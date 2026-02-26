from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph ,START , END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import operator
from typing import TypedDict


class AgentState(TypedDict):
    topic: str
    audience: str
    tone: str
    blog_kind: str
    mode : str
    needs_research : bool
    queries : List[str]
    evidence : List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int,str]], operator.add]  # worker 4 , section md 

    merged_md : str
    md_with_placeholders : str
    image_specs : List[dict]


    final: str

class Task(BaseModel):
    id : int
    title: str
    goal : str = Field(...,
                       description="One sentence describing what the reader should be able to do/understand after this section.")
    bullets : list[str] = Field(...,
                                min_length=3, max_length=5,
                                description="3-5 bullet points describing the key takeaways of this section."
                                "Each bullet point should be a single sentence and should not exceed 20 words.")
    target_words : int = Field(...,
                                description="The target word count for this section. Should be between 150 and 500 words.")
    section_type : Literal["introduction", "core","examples","checklist", "common_mistakes", "body", "conclusion"] = Field(...,
                                                        description="The type of the section. Can be 'introduction', 'body', or 'conclusion'."
                                                                    "Use coomon mistakes exactly once in the plan if required .")
    tags : List[str] = Field(default_factory=list)
    requires_research : bool = False
    requires_citations : bool = False
    requires_code : bool = False
    code_language : Optional[str] = None
    code_description : Optional[str] = None




class Plan(BaseModel):
    blog_title : str
    audience : str = Field(...,
                        description="A one sentence description of the target audience for this blog post. This should include information about the reader's background, interests, and what they hope to gain from reading the blog post.")
    tone : str = Field(...,
                        description="A one sentence description of the tone and style of the blog post. This should include information about the desired level of formality, use of humor, and overall writing style that should be used in the blog post." \
                        "For example, the tone could be described as 'conversational and engaging with a touch of humor' or 'professional and informative with a formal tone'." \
                        "The tone should be consistent throughout the blog post and should align with the target audience and the topic of the blog post.")
    blog_kind : Literal["how-to", "listicle", "opinion", "case-study", "news", "interview", "explainer", "System_design"] = "explainer"
    constraints : List[str] = Field(default_factory=list)
    task: list[Task]



class EvidenceItem(BaseModel):
    title : str
    url : str
    published_at : Optional[str] = None
    snippet : Optional[str] = None
    source : Optional[str] = None


class RouterDecision(BaseModel):
    needs_research : bool 
    mode : Literal["write", "research", "code","closed_book","hybrid","open_book"] 
    queries : List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence_items : List[EvidenceItem] = Field(default_factory=list)



class ImageSpec(BaseModel):
    placeholder : str = Field(...,description="e.g : [[IMAGE_1]]")
    file_name: str = Field(...,description="Save under Images / e.g : image1.png")
    alt : str 
    caption : str
    prompt : str = Field(...,description="Prompt to send to the Image Model")
    size : Literal["256x256", "512x512", "1024x1024"] = "512x512"
    quality : Literal["standard", "ultra"] = "standard"

class GlobalImagePlan(BaseModel):
    md_with_placeholders : str
    images : List[ImageSpec] = Field(default_factory=list)