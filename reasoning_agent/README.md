# 🤖 Reasoning Agent: AI-Powered Blog Generation System

A sophisticated autonomous AI agent that generates high-quality technical blog posts using multi-step reasoning, web research, and content orchestration. Built with LangGraph, LangChain, and advanced LLM capabilities.

## 🎯 Overview

Reasoning Agent is an intelligent content generation system designed to produce professional, research-backed blog posts on technical topics. It leverages a graph-based agentic architecture to coordinate multiple specialized AI workers—from research specialists to content writers to image planners—creating cohesive, well-structured narratives from topic briefs.

### Key Capabilities

- **Intelligent Routing**: Automatically determines if research is needed based on topic complexity
- **Web Research Integration**: Uses Tavily Search to gather current, authoritative sources
- **Multi-Worker Orchestration**: Coordinates specialized agents for planning, writing, and image generation
- **Structured Content Generation**: Produces blogs with sections, research citations, and code examples
- **Image Planning**: Generates detailed image specifications for visual enhancement
- **Error Handling**: Robust error recovery and progress tracking
- **Interactive UI**: Streamlit-based interface for real-time monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for LLM inference)
- Tavily API key (for web search)
- Google Generative AI API key (optional, for image generation)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mehraj-alom/Agent
   cd reasoning_agent
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   TAVILY_API_KEY=your_tavily_key_here
   BYTEZ_API_KEY=your_bytez_key_here  # Optional
   BOT=your_bot_identifier
   ```

### Usage

#### Via CLI
```python
from reasoning_agent.Core.core import run
from datetime import date

result = run(
    topic="Machine Learning in 2026",
    audience="Software engineers interested in AI/ML",
    tone="professional and informative",
    blog_kind="explainer",
    as_of=date.today().isoformat()
)

print(result['final'])  # Generated blog content
```

#### Via Streamlit Web App
```bash
streamlit run streamlit_app.py
```

The web interface provides:
- Interactive form for blog parameters
- Real-time progress tracking with status updates
- Section-by-section content display
- Image generation and management
- Error handling and recovery

## 📁 Project Structure

```
reasoning_agent/
├── Core/
│   ├── core.py              # Main agentic orchestration engine
│   ├── core_models.py       # Pydantic data models and type definitions
│   └── __init__.py
├── Config/
│   └── config.py            # Configuration settings
├── Notebooks/
│   └── 01_notebook.ipynb    # Jupyter notebook for exploration
├── requirements.txt         # Python dependencies
├── LICENSE                  # Apache 2.0 License
└── README.md               # This file
```

## 🏗️ Architecture

### Agent Workers

The system employs a LangGraph-based architecture with specialized workers:

1. **Router** - Analyzes the topic and determines if research is needed
2. **Researcher** - Conducts web searches using Tavily API
3. **Planner** - Creates detailed blog structure and task breakdown
4. **Content Writers** - Generate individual sections with citations
5. **Image Planner** - Designs visual elements with detailed prompts
6. **Orchestrator** - Coordinates the entire pipeline

### Data Flow

```
User Input (Topic, Audience, Tone)
         ↓
    Router Decision
         ↓
   Research (if needed)
         ↓
    Planning Phase
         ↓
  Parallel Section Writing
         ↓
  Image Specification
         ↓
  Content Assembly
         ↓
  Final Output
```

## 📋 Core Models

### AgentState
Central state management object that tracks:
- Topic, audience, tone, blog type
- Research queries and evidence
- Section content and metadata
- Image specifications
- Final generated content

### Plan
Defines the blog structure with:
- Blog title and target audience
- Tone and writing style
- Blog type (how-to, listicle, opinion, case-study, etc.)
- Task breakdown with:
  - Section titles and goals
  - Key takeaways (bullet points)
  - Word count targets
  - Content type (intro, core, examples, etc.)
  - Code requirements and citations

### EvidenceItem
Research findings with:
- Title and source URL
- Publication date
- Content snippet
- Source attribution

## 🔧 Configuration

Key environment variables:
- `OPENAI_API_KEY` - OpenAI API authentication
- `TAVILY_API_KEY` - Web search API credentials
- `BYTEZ_API_KEY` - Image generation service (optional)
- `BOT` - Bot identifier for tracking

## 📦 Dependencies

Core dependencies:
- **langgraph** - Graph-based agent orchestration
- **langchain** & **langchain-openai** - LLM framework
- **langchain-tavily** - Web search integration
- **langchain-community** - Community integrations
- **langchain-google-genai** - Google AI integration
- **streamlit** - Web UI framework
- **pydantic** - Data validation
- **duckduckgo-search** - Alternative search provider
- **python-dotenv** - Environment configuration

See `requirements.txt` for complete list with versions.

## 🧪 Testing

Run error handling tests to verify the pipeline:
```bash
python test_error_handling.py
```

This validates:
- Error capturing and reporting
- Progress callback functionality
- Result structure integrity
- Pipeline success metrics

## 🎨 Features in Detail

### Intelligent Research Routing
The router automatically decides whether web research is needed based on:
- Topic novelty and recency requirements
- Complexity assessment
- Citation needs

### Multi-Mode Generation
Supports different generation modes:
- **closed_book** - Knowledge-based generation without research
- **open_book** - Research-augmented generation
- **hybrid** - Combined approach
- **write** - Direct content generation
- **research** - Research-focused mode
- **code** - Code-focused content

### Progress Tracking
Real-time progress updates with emoji indicators:
- 📊 Analysis phases
- 🔍 Research operations
- 📝 Writing stages
- 🎨 Image generation
- ✅ Completion milestones

## 🛠️ Development

### Adding Custom Workers

Extend the agent by adding new workers to `core.py`:

```python
def my_custom_worker(state: AgentState) -> dict:
    """Custom worker implementation"""
    emit_progress("🔧 Running custom worker")
    # Your logic here
    return {"updated_field": value}

# Add to graph
graph.add_node("my_worker", my_custom_worker)
graph.add_edge("my_worker", END)
```

### Debugging
Enable detailed logging by setting progress callbacks:
```python
def my_callback(message: str):
    print(f"[LOG] {message}")

run(..., progress_callback=my_callback)
```

## 📄 Output Structure

Generated blog includes:
- Structured markdown content
- Section metadata
- Research citations
- Code examples (when requested)
- Image specifications with alt text
- SEO-optimized formatting

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional language support
- More blog type templates
- Enhanced image generation
- Performance optimization
- New research backends

## 📜 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with cutting-edge technologies:
- LangChain & LangGraph for agent orchestration
- OpenAI for language models
- Tavily for web search capabilities
- Streamlit for the web interface

## 📮 Support

For issues, questions, or suggestions:
1. Check existing documentation in the Notebooks folder
2. Review test cases in `test_error_handling.py`
3. Consult the core models in `Core/core_models.py` for data structures

---

**Built for technical content creators and developers who demand high-quality, research-backed blog generation.**
