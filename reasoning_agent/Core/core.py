from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph ,START , END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import operator
from reasoning_agent.Core.core_models import AgentState, Plan, Task , EvidenceItem , EvidencePack, RouterDecision , GlobalImagePlan , ImageSpec
from typing import TypedDict
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from pathlib import Path
import os


env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


parent_env_path = Path(__file__).parent.parent.parent / ".env"
if parent_env_path.exists():
    load_dotenv(parent_env_path)

import json
from datetime import date, datetime , timedelta
from bytez import Bytez

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BYTEZ_API_KEY = os.getenv("BYTEZ_API_KEY")
Bot = os.getenv("BOT")
# Search Engine for research worker
engine = TavilySearch(max_results=2, api_key=TAVILY_API_KEY,
                             search_depth = "advanced")

# Language model for all workers
model = ChatOpenAI(
    model="qwen/qwen3-vl-235b-a22b-thinking",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY,
    max_tokens=8000
)

# Progress callback for UI updates
progress_callback = None

def set_progress_callback(callback):
    """Set a callback function to receive progress updates"""
    global progress_callback
    progress_callback = callback

def emit_progress(message: str):
    """Emit a progress message to the UI"""
    global progress_callback
    if progress_callback:
        try:
            progress_callback(message)
        except Exception as e:
            # If callback fails (e.g., in background thread), just log to console
            print(f"[Progress] {message}")

# model = OpenAI(
#   base_url="https://openrouter.ai/api/v1",
#   api_key=Bot
# )


sdk = Bytez(BYTEZ_API_KEY)
Imag_gen_model = sdk.model("google/imagen-4.0-ultra-generate-001")



ROUTER_SYSTEM = """You are a routing module for a high experience blog planner.
Decide whether web research is needed BEFORE planning.
Modes:
closed_book (needs_research=false):
- Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
hybrid (needs_research=true):
- Mostly evergreen but needs up-to-date examples/tools/models to be useful.
open_book (needs_research=true):
- Mostly volatile: weekly roun ups, "this week", "latest", rankings, pricing, policy/regulation.
If needs_research=true:
Output 3-10 high-signal queries.
Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.

IMPORTANT: Respond with valid JSON only:
{
  "needs_research": true/false,
  "mode": "closed_book" | "hybrid" | "open_book",
  "queries": ["query1", "query2"]
}
"""


def router(state : AgentState) -> dict :
    topic = state['topic']
    
    emit_progress("📊 Analyzing Topic")
    emit_progress("🤖 Running Router")
    emit_progress("📈 Evaluating if research is needed...")
    
    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {topic}"),
    ]
    response = model.invoke(messages)
    
    emit_progress("✔️ Router decision complete")
    
    try:
        json_str = response.content.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        data = json.loads(json_str)
        return {
            "needs_research": data.get("needs_research", False),
            "mode": data.get("mode", "closed_book"),
            "queries": data.get("queries", [])    
        }
    except Exception as e:
        print(f"Router error: {e}")
        return {"needs_research": False, "mode": "closed_book", "queries": []}



def route_next (state : AgentState) -> str:
    return "research" if state['needs_research'] else "orchestrator"



###########
# Research function for research worker
###########
def tavily_search(quary:str , max_results : int = 5) -> List[EvidenceItem] :
    engine = TavilySearch(max_results=max_results, api_key=TAVILY_API_KEY,
                             search_depth = "advanced")
    result = engine.invoke(quary)
    evidence_items = []
    for item in result['results']:
        evidence_items.append(EvidenceItem(
            title=item['title'],
            url=item['url'],
            published_at=item.get('published_at'),
            snippet=item.get('content'),
            source=item.get('source')
        ))
    return evidence_items



###########
# Research synthesis function for research worker
###########
RESEARCH_SYSTEM = """You are a research synthesizer for technical and Skilled Blog writing.

Given raw web search results, produce a deduplicated list.
Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
- If missing or unclear, set published_at = null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL

IMPORTANT: Return valid JSON only:
{
  "evidence_items": [
    {"title": "...", "url": "...", "published_at": "2026-02-25" or null, "snippet": "...", "source": "..."}
  ]
}
"""

def research(state : AgentState) -> dict:
    queries = state.get('queries', [])
    max_results_per_query = 6
    raw_results : List[dict] = []

    emit_progress("🔍 Starting Research")
    emit_progress("🌐 Querying Search Engine")
    
    try:
        for i, query in enumerate(queries, 1):
            emit_progress(f"📰 Visiting Source {i}/{len(queries)}")
            raw_results.extend(tavily_search(query, max_results=max_results_per_query))

        if not raw_results:
            emit_progress("⚠️ No results found")
            return {"evidence": []}
        
        emit_progress("🧹 Deduplicating Results")
        
        emit_progress("📊 Processing search results")
        response = model.invoke(
            [
                SystemMessage(content=RESEARCH_SYSTEM),
                HumanMessage(content=f"Raw search results: \n{raw_results}")
            ]
        )
        json_str = response.content.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        data = json.loads(json_str)
        evidence_items = [EvidenceItem(**item) for item in data.get("evidence_items", [])]
        dedup = {}
        for item in evidence_items:
            if item.url and item.url not in dedup:
                dedup[item.url] = item
        
        emit_progress(f"✅ Research Complete - Found {len(dedup)} sources")
        return {"evidence": list(dedup.values())}
    except Exception as e:
        print(f"Research error: {e}")
        emit_progress(f"⚠️ Research encountered an error: {str(e)}")
        return {"evidence": []}


############
# Orchestrator function for planning worker
############

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
Create 10 to 15 sections (tasks) suitable for the topic and audience.
Each task must include:
1) goal (1 sentence)
2) 3-6 bullets that are concrete, specific, and non-overlapping
3) target word count (200 - 500)
4) section_type MUST be one of: 'introduction', 'core', 'examples', 'checklist', 'common_mistakes', 'body', 'conclusion'

IMPORTANT: Return valid JSON only:
{
  "blog_title": "...",
  "audience": "...",
  "tone": "...",
  "blog_kind": "explainer",
  "constraints": [],
  "task": [
    {
      "id": 1,
      "title": "...",
      "goal": "...",
      "bullets": ["...", "..."],
      "target_words": 500,
      "section_type": "introduction",
      "tags": [],
      "requires_research": false,
      "requires_citations": false,
      "requires_code": false,
      "code_language": null,
      "code_description": null
    }
  ]
}
"""


def orchestrator(state : AgentState) -> dict:
    topic = state['topic']
    audience = state['audience']
    tone = state['tone']
    blog_kind = state['blog_kind']
    evidence = state.get('evidence', [])
    mode = state['mode']

    emit_progress("📋 Creating Blog Plan")
    emit_progress("🎯 Analyzing Structure")

    messages = [
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=f"Topic: {topic}\nAudience: {audience}\nTone: {tone}\nBlog Kind: {blog_kind}\nMode: {mode}\nEvidence: {evidence}")
    ]

    try:
        emit_progress("💭 Generating Sections")
        response = model.invoke(messages)
        json_str = response.content.strip()
        
        # Try multiple ways to extract JSON
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        # Find JSON object if it's embedded in text
        if not json_str.startswith("{"):
            start_idx = json_str.find("{")
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(json_str)):
                    if json_str[i] == "{":
                        brace_count += 1
                    elif json_str[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = json_str[start_idx:i+1]
                            break
        
        data = json.loads(json_str)
        
        # Normalize section_type to valid values
        for task in data.get("task", []):
            section_type = task.get("section_type", "body").lower()
            valid_types = ['introduction', 'core', 'examples', 'checklist', 'common_mistakes', 'body', 'conclusion']
            if section_type not in valid_types:
                # Map common types to valid ones
                type_mapping = {
                    'intro': 'introduction',
                    'how-to': 'core',
                    'tutorial': 'core',
                    'best-practices': 'core',
                    'tips': 'examples',
                    'summary': 'conclusion',
                    'wrap-up': 'conclusion',
                }
                task["section_type"] = type_mapping.get(section_type, "body")
            else:
                task["section_type"] = section_type
        
        emit_progress("✅ Plan Created")
        plan = Plan(**data)
        return {"plan": plan}
    except Exception as e:
        print(f"Orchestrator error: {e}")
        print(f"Response content: {response.content[:200] if 'response' in locals() else 'No response'}")
        emit_progress(f"⚠️ Plan generation failed: {str(e)}")
        # Create a minimal valid plan with at least one fallback task
        fallback_task = Task(
            id=1,
            title="Overview",
            goal=f"Provide an overview of {topic}",
            bullets=[
                "Main concepts and definitions",
                "Practical applications",
                "Key takeaways"
            ],
            target_words=500,
            section_type="body",
            tags=[],
            requires_research=False,
            requires_citations=False,
            requires_code=False,
            code_language=None,
            code_description=None
        )
        return {"plan": Plan(blog_title=f"{topic} Guide", audience=audience, tone=tone, blog_kind=blog_kind, task=[fallback_task])}

#########
# Fanout function to send each section to a different worker
#########

def fanout(state : AgentState) -> dict:
    plan = state.get('plan')
    if not plan or not plan.task:
        emit_progress("⚠️ No sections to generate")
        # Return a default message
        return [
            Send(
                "worker",{
                    "task" : {
                        "id": 1,
                        "title": "Overview",
                        "goal": f"Provide an overview of {state['topic']}",
                        "bullets": ["Main concept", "Applications", "Summary"],
                        "target_words": 300,
                        "section_type": "body",
                        "tags": [],
                        "requires_research": False,
                        "requires_citations": False,
                        "requires_code": False,
                        "code_language": None,
                        "code_description": None
                    },
                    "topic" : state['topic'],
                    "mode" : state.get('mode', ''),
                    "plan" : plan.model_dump() if plan else {},
                    "evidence" : [item.model_dump() for item in state.get('evidence',[])]
                }
            )
        ]
    
    return [
        Send(
            "worker",{
                "task" : task.model_dump(),
                "topic" : state['topic'],
                "mode" : state.get('mode', ''),
                "plan" : state["plan"].model_dump(),
                "evidence" : [item.model_dump() for item in state.get('evidence',[])]
            }
        )
        for task in state['plan'].task
    ]



###########
# Worker function to write each section
###########

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker(payload: dict) -> dict:
    
    try:
        task = Task(**payload["task"])
        plan = Plan(**payload["plan"])
        evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
        topic = payload["topic"]
        mode = payload.get("mode", "closed_book")

        emit_progress(f"✍️ Writing: {task.title}")

        bullets_text = "\n- " + "\n- ".join(task.bullets)

        evidence_text = ""
        if evidence:
            evidence_text = "\n".join(
                f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
                for e in evidence[:20]
            )

        section_md = model.invoke(
            [
                SystemMessage(content=WORKER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog title: {plan.blog_title}\n"
                        f"Audience: {plan.audience}\n"
                        f"Tone: {plan.tone}\n"
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Constraints: {plan.constraints}\n"
                        f"Topic: {topic}\n"
                        f"Mode: {mode}\n\n"
                        f"Section title: {task.title}\n"
                        f"Goal: {task.goal}\n"
                        f"Target words: {task.target_words}\n"
                        f"Tags: {task.tags}\n"
                        f"requires_research: {task.requires_research}\n"
                        f"requires_citations: {task.requires_citations}\n"
                        f"requires_code: {task.requires_code}\n"
                        f"Bullets:{bullets_text}\n\n"
                        f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                    )
                ),
            ]
        ).content.strip()

        return {"sections": [(task.id, section_md)]}
    except Exception as e:
        print(f"Worker error processing task: {e}")
        # Don't call emit_progress here - it can fail in background threads
        # Just log to console
        task_id = payload.get("task", {}).get("id", 1)
        task_title = payload.get("task", {}).get("title", "Section")
        fallback_md = f"## {task_title}\n\nThis section could not be generated. Please try again.\n"
        return {"sections": [(task_id, fallback_md)]}



#########
# Reducer function to combine all sections into a final markdown
#########

# def reducer(state: AgentState) -> dict:

#     plan = state["plan"]

#     ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
#     body = "\n\n".join(ordered_sections).strip()
#     final_md = f"# {plan.blog_title}\n\n{body}\n"

#     filename = f"{plan.blog_title}.md"
#     Path(filename).write_text(final_md, encoding="utf-8")

#     return {"final": final_md}

def merge_content(state: AgentState) -> dict:

    try:
        emit_progress("🔗 Merging Content")
        plan = state["plan"]
        
        if not plan:
            emit_progress("⚠️ No plan available")
            return {"merged_md": "# Blog\n\nNo content to merge."}

        ordered_sections = [md for _, md in sorted(state.get("sections", []), key=lambda x: x[0])]
        
        if not ordered_sections:
            emit_progress("⚠️ No sections available")
            body = "No sections were generated."
        else:
            body = "\n\n".join(ordered_sections).strip()
        
        merged_md = f"# {plan.blog_title}\n\n{body}\n"
        
        emit_progress("✅ Content Merged")
        return {"merged_md": merged_md}
    except Exception as e:
        print(f"Merge content error: {e}")
        emit_progress(f"⚠️ Error merging content: {str(e)}")
        return {"merged_md": "# Blog\n\nError during content merging. Please try again."}



DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: return md_with_placeholders = input markdown and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.

For each image, provide a detailed visual description prompt.
Return JSON with md_with_placeholders and image prompts array.
"""

def decide_images(state: AgentState) -> dict:
    try:
        merged_md = state.get("merged_md", "")
        plan = state.get("plan")
        
        if not plan:
            emit_progress("⚠️ No plan for image decision")
            return {"md_with_placeholders": merged_md, "image_specs": []}

        assert plan is not None

        emit_progress("🖼️ Planning Images")

        image_specs = []
        md_with_placeholders = merged_md
        
        if plan.blog_kind in ["explainer", "how-to", "tutorial"]:

            emit_progress("📐 Generating Image Specs")

            md_with_placeholders = merged_md.replace(
                "# " + plan.blog_title,
                "# " + plan.blog_title + "\n\n[[IMAGE_1]]"
            )
            
            image_specs = [
                {
                    "placeholder": "[[IMAGE_1]]",
                    "file_name": "main_concept.png",
                    "alt": f"Illustration of {plan.blog_title}",
                    "caption": f"Visual overview of {plan.blog_title}",
                    "prompt": f"Create a clear, professional technical diagram or illustration explaining {state.get('topic', plan.blog_title)}. Use modern design, clear labels, and ensure it's suitable for a technical blog post.",
                    "size": "1024x1024",
                    "quality": "standard"
                }
            ]
        
        emit_progress("✅ Image Plan Ready")
        return {"md_with_placeholders": md_with_placeholders, "image_specs": image_specs}
    except Exception as e:
        print(f"Decide images error: {e}")
        emit_progress(f"⚠️ Error in image planning: {str(e)}")
        merged_md = state.get("merged_md", "")
        return {"md_with_placeholders": merged_md, "image_specs": []}



def Generate_images(image_specs: List[dict]) -> List[dict]:
    """Generate images using Bytez and return list with placeholder + URL"""
    generated_images = []
    for i, spec in enumerate(image_specs, 1):
        try:
            emit_progress(f"🎨 Generating image {i}/{len(image_specs)}")
            response = Imag_gen_model.run(spec["prompt"])
            image_url = response.output
            generated_images.append({
                "placeholder": spec["placeholder"],
                "url": image_url,
                "caption": spec.get("caption", ""),
                "alt": spec.get("alt", "")
            })
            emit_progress(f"✅ Image {i} Complete")
        except Exception as e:
            print(f"Error generating image for {spec['placeholder']}: {e}")
            emit_progress(f"⚠️ Image {i} failed")
            generated_images.append({
                "placeholder": spec["placeholder"],
                "url": None,
                "error": str(e),
                "caption": spec.get("caption", ""),
                "alt": spec.get("alt", "")
            })
    return generated_images


def generate_and_place_images(state: AgentState) -> dict:

    try:
        emit_progress("📸 Processing Generated Images")

        plan = state.get("plan")
        
        if not plan:
            emit_progress("⚠️ No plan available")
            return {"final": state.get("md_with_placeholders", "")}

        md = state.get("md_with_placeholders") or state.get("merged_md", "")
        image_specs = state.get("image_specs", []) or []

       
        if not image_specs:
            emit_progress("📝 Finalizing Blog")
            filename = f"{plan.blog_title}.md"
            try:
                Path(filename).write_text(md, encoding="utf-8")
            except Exception as e:
                print(f"Error writing file: {e}")
            return {"final": md}

        generated_images = Generate_images(image_specs)

        emit_progress("🔗 Embedding Images in Content")
        
        for img in generated_images:
            placeholder = img["placeholder"]
            
            if img.get("url"):
                
                img_md = f"![{img['alt']}]({img['url']})\n*{img['caption']}*"
            else:
                img_md = (
                    f"> **[IMAGE GENERATION FAILED]**\n>\n"
                    f"> *{img['caption']}*\n>\n"
                    f"> **Alt:** {img['alt']}\n>\n"
                    f"> **Error:** {img.get('error', 'Unknown error')}\n"
                )
            
            md = md.replace(placeholder, img_md)

        emit_progress("💾 Saving to Disk")
        filename = f"{plan.blog_title}.md"
        try:
            Path(filename).write_text(md, encoding="utf-8")
        except Exception as e:
            print(f"Error writing file: {e}")
        emit_progress("✅ Blog Complete!")
        return {"final": md}
    except Exception as e:
        print(f"Generate and place images error: {e}")
        emit_progress(f"⚠️ Error generating images: {str(e)}")
        md = state.get("md_with_placeholders") or state.get("merged_md", "")
        return {"final": md}

#########
# Define the state graph and compile the app
#########
reducer_graph = StateGraph(AgentState)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)

reducer_subgraph = reducer_graph.compile()


graph = StateGraph(AgentState)
graph.add_node("router", router)
graph.add_node("research", research)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer_subgraph)
graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
graph.add_edge("research", "orchestrator")

graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

app = graph.compile()   


# -----------------------------
def run(topic: str, audience: str = "Developers", tone: str = "professional", blog_kind: str = "explainer", as_of: Optional[str] = None, progress_callback=None):
    """Execute the blog generation pipeline."""
    if as_of is None:
        as_of = date.today().isoformat()

    # Set the progress callback if provided
    if progress_callback:
        set_progress_callback(progress_callback)


    try:
        out = app.invoke(
            {
                "topic": topic,
                "audience": audience,
                "tone": tone,
                "blog_kind": blog_kind,
                "mode": "",
                "needs_research": False,
                "queries": [],
                "evidence": [],
                "plan": None,
                "as_of": as_of,
                "recency_days": 7,
                "sections": [],
                "merged_md": "",
                "md_with_placeholders": "",
                "image_specs": [],
                "final": "",
            }
        )

        return out
    except Exception as e:
        import traceback
        exc_type = type(e).__name__
        exc_str = str(e) if str(e) else f"({exc_type} with no message)"
        error_msg = f"Pipeline execution failed ({exc_type}): {exc_str}"
        
        #
        print(f"\n{'='*60}")
        print(f"PIPELINE ERROR DETAILS")
        print(f"{'='*60}")
        print(f"Error Type: {exc_type}")
        print(f"Error Message: {exc_str}")
        print(f"\nFull Traceback:")
        print(traceback.format_exc())
        print(f"{'='*60}\n")
        
        emit_progress(f"❌ {error_msg}")
        
       
        return {
            "topic": topic,
            "audience": audience,
            "tone": tone,
            "blog_kind": blog_kind,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of,
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
            "error": error_msg
        }


if __name__ == "__main__":
    run("Self Attention in Transformer Architecture and The Evalution of this paper till today")