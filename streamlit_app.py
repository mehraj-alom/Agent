import streamlit as st
import sys
from pathlib import Path
from datetime import date

# Add the reasoning_agent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from reasoning_agent.Core.core import run
from reasoning_agent.Core.core_models import Plan

# Configure page
st.set_page_config(
    page_title="Blog Generator Agent",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .section-header {
        font-size: 1.5em;
        color: #1f77b4;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown('<h1 class="main-header">📝 Intelligent Blog Generator</h1>', unsafe_allow_html=True)
with col2:
    st.markdown("**v1.0**")

st.markdown("Generate high-quality technical blog posts with AI-powered planning, research, and content generation.")
st.divider()

# Sidebar for configuration
st.sidebar.markdown('<h2 class="section-header">Configuration</h2>', unsafe_allow_html=True)

# Input fields
topic = st.sidebar.text_input(
    "📌 Blog Topic",
    value="Self Attention in Transformer Architecture",
    help="Enter the main topic for your blog post"
)

audience = st.sidebar.text_input(
    "👥 Target Audience",
    value="Software engineers and machine learning developers",
    help="Describe who will read this blog"
)

tone = st.sidebar.text_input(
    "🎯 Tone & Style",
    value="Professional yet accessible, with practical examples",
    help="How should the writing feel?"
)

blog_kind = st.sidebar.selectbox(
    "📖 Blog Type",
    options=["explainer", "how-to", "tutorial", "opinion", "case-study", "news", "interview", "listicle"],
    index=0,
    help="What type of blog post is this?"
)

st.sidebar.divider()

# Generate button
col1, col2 = st.sidebar.columns(2)
with col1:
    generate_btn = st.sidebar.button("🚀 Generate Blog", use_container_width=True)
with col2:
    clear_btn = st.sidebar.button("🔄 Clear", use_container_width=True)

# Main content area
if clear_btn:
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

if generate_btn:
    # Create placeholders for progress
    with st.container():
        st.markdown('<h2 class="section-header">⚙️ Generating Blog...</h2>', unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        details_container = st.container()
        
        # Detailed steps list
        steps = [
            ("📊 Analyzing Topic", "Examining your topic for research requirements..."),
            ("🤖 Running Router", "Determining if research is needed..."),
            ("📈 Evaluating Mode", "Deciding between closed-book, hybrid, or open-book approach..."),
            
            ("🔍 Starting Research", "Initiating web search for evidence..."),
            ("🌐 Querying Search Engine", "Searching for relevant information..."),
            ("📰 Visiting Source 1", "Gathering data from authoritative sources..."),
            ("📄 Visiting Source 2", "Collecting additional references..."),
            ("📑 Visiting Source 3", "Finding supporting evidence..."),
            ("🧹 Deduplicating Results", "Removing duplicate sources and organizing findings..."),
            ("✅ Research Complete", "Successfully gathered all evidence..."),
            
            ("📋 Planning Structure", "Creating blog outline..."),
            ("🎯 Generating Plan", "Orchestrating section breakdown..."),
            ("📝 Defining Sections", "Creating 5-6 detailed sections..."),
            ("🏷️ Adding Metadata", "Setting section types and requirements..."),
            ("✔️ Validating Plan", "Ensuring plan matches schema..."),
            
            ("✍️ Generating Content", "Writing blog sections..."),
            ("📄 Writing Section 1", "Generating introduction and overview..."),
            ("📄 Writing Section 2", "Creating core concepts section..."),
            ("📄 Writing Section 3", "Adding examples and practical applications..."),
            ("📄 Writing Section 4", "Documenting best practices and tips..."),
            ("📄 Writing Section 5", "Writing conclusion and next steps..."),
            ("📄 Writing Section 6", "Adding final recommendations..."),
            ("🔗 Merging Content", "Combining all sections into cohesive blog..."),
            
            ("🖼️ Planning Images", "Deciding which images would enhance understanding..."),
            ("🎨 Generating Image Prompts", "Creating detailed visual descriptions..."),
            ("🖌️ Calling Image Generator", "Sending prompts to Bytez model..."),
            ("⏳ Rendering Image 1", "AI is generating technical diagram..."),
            ("✨ Finalizing Image", "Optimizing and embedding image..."),
            ("🔗 Embedding Images", "Inserting images into markdown..."),
            
            ("📦 Finalizing Blog", "Preparing final output..."),
            ("💾 Writing to File", "Saving blog post to disk..."),
            ("✅ Complete", "Blog generation successful!"),
        ]
        
        total_steps = len(steps)
        
        # Progress callback to receive real updates from agent
        progress_messages = []
        def progress_callback(message: str):
            """Callback to receive progress updates from the agent"""
            progress_messages.append(message)
            try:
                with details_container:
                    st.caption(f"🔄 {message}")
            except Exception as e:
                # Streamlit UI updates may fail in background threads
                # Just log the message instead
                print(f"Progress: {message}")
        
        try:
            # Run the agent with progress callback
            status_text.text("🚀 Starting blog generation pipeline...")
            progress_bar.progress(0.6,"About to Complete")
            
            result = run(
                topic=topic,
                audience=audience,
                tone=tone,
                blog_kind=blog_kind,
                as_of=date.today().isoformat(),
                progress_callback=progress_callback
            )
            
            # Check if there was an error
            if result.get("error"):
                progress_bar.progress(100)
                status_text.markdown("### ❌ Error")
                error_msg = result.get('error', 'Unknown error')
                st.error(f"❌ Error during generation:\n\n{error_msg}")
                st.info("💡 Tip: Check your API keys and internet connection, then try again.\n\nCheck the terminal output for detailed error information.")
                st.stop()
            
            progress_bar.progress(100)
            status_text.markdown("### ✅ Complete")
            with details_container:
                st.caption("✨ Blog generation successful!")
            
            st.markdown('<div class="success-box"><strong>✨ Success!</strong> Your blog post has been generated successfully.</div>', unsafe_allow_html=True)
            
            # Display results
            st.markdown('<h2 class="section-header">📄 Generated Blog Post</h2>', unsafe_allow_html=True)
            
            # Get the final markdown
            final_md = result.get("final", "")
            
            if final_md:
                # Display the markdown
                st.markdown(final_md)
                
                # Download button
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📥 Download as Markdown",
                        data=final_md,
                        file_name=f"{topic.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                
                # Save confirmation
                plan = result.get('plan')
                if plan:
                    if isinstance(plan, dict):
                        plan_title = plan.get('blog_title', 'Blog')
                    else:
                        plan_title = getattr(plan, 'blog_title', 'Blog')
                    file_path = Path(f"{plan_title}.md")
                    if file_path.exists():
                        with col2:
                            st.info(f"✅ Saved to: `{file_path}`")
                
                # Statistics
                with col3:
                    word_count = len(final_md.split())
                    st.metric("Word Count", f"{word_count:,}")
            
            else:
                st.warning("⚠️ No content generated. Please check the configuration and try again.")
        
        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
            st.info("💡 Tip: Check your API keys and internet connection, then try again.")

else:
    # Welcome message
    st.markdown('<h2 class="section-header">🎯 How to Use</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Step 1: Configure Your Blog
        - **Topic**: What's your blog about?
        - **Audience**: Who are the readers?
        - **Tone**: What's the writing style?
        - **Type**: Choose the blog format
        """)
    
    with col2:
        st.markdown("""
        ### Step 2: Generate
        - Click the **Generate Blog** button
        - The system will:
          - Analyze your topic
          - Research and gather evidence
          - Generate structured sections
          - Create and embed images
        """)
    
    st.divider()
    
    st.markdown('<h2 class="section-header">✨ Features</h2>', unsafe_allow_html=True)
    
    feature_cols = st.columns(4)
    
    with feature_cols[0]:
        st.markdown("""
        **🤖 AI Planning**
        
        Intelligent outline generation tailored to your topic and audience
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **🔍 Research**
        
        Automated web search and evidence gathering
        """)
    
    with feature_cols[2]:
        st.markdown("""
        **✍️ Content Generation**
        
        High-quality sections with proper structure and depth
        """)
    
    with feature_cols[3]:
        st.markdown("""
        **🖼️ Image Generation**
        
        AI-generated technical diagrams and illustrations
        """)
    
    st.divider()
    
    # Example configurations
    st.markdown('<h2 class="section-header">📚 Example Topics</h2>', unsafe_allow_html=True)
    
    examples = [
        {
            "topic": "Machine Learning Model Deployment",
            "audience": "DevOps engineers and ML engineers",
            "tone": "Technical and practical"
        },
        {
            "topic": "Python Async Programming",
            "audience": "Backend developers",
            "tone": "Educational with code examples"
        },
        {
            "topic": "Database Optimization Strategies",
            "audience": "Database administrators and developers",
            "tone": "Professional and comprehensive"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        with st.expander(f"Example {i}: {example['topic']}"):
            st.write(f"**Audience**: {example['audience']}")
            st.write(f"**Tone**: {example['tone']}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>🚀 Powered by LangGraph, OpenAI, and Bytez Image Generation</p>
    <p>© 2026 Intelligent Blog Generator | All rights reserved</p>
</div>
""", unsafe_allow_html=True)
