"""
Streamlit UI for Hybrid Financial Analyst Chatbot

This is the web interface for the chatbot, providing:
- Chat interface with message history
- Source traceability (SQL queries or document citations)
- Clear indication of which data source was used
- Conversation management (clear history)

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from database import FinancialDatabase
from rag_pipeline import RAGPipeline
from llm_handler import LLMHandler
from chatbot import FinancialChatbot


# Page configuration
st.set_page_config(
    page_title="Financial Analyst Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    .sql-query {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
    .message-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .message-assistant {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_chatbot():
    """
    Initialize chatbot components

    Uses @st.cache_resource to initialize only once and reuse across sessions
    This is critical for performance - loading embeddings takes time!
    """
    try:
        # Initialize database
        csv_path = "./data/financial_data.csv"
        db = FinancialDatabase(csv_path)

        # Initialize RAG pipeline
        docs_dir = "./data/unstructured"
        rag = RAGPipeline(docs_directory=docs_dir)

        # Build vector store if not already built
        if rag.collection.count() == 0:
            st.info("🔄 Building vector store for the first time... This may take a minute.")
            rag.build_vector_store()
            st.success("✅ Vector store built successfully!")

        # Initialize LLM handler
        llm = LLMHandler()

        # Initialize chatbot
        chatbot = FinancialChatbot(database=db, rag_pipeline=rag, llm_handler=llm)

        return chatbot

    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        st.info("Please make sure:\n1. Your .env file contains GOOGLE_API_KEY\n2. Data files are in ./data/ directory")
        st.stop()


def display_message(role, content, metadata=None):
    """
    Display a chat message with styling

    Args:
        role: "user" or "assistant"
        content: Message content
        metadata: Optional metadata (sources, route, etc.)
    """
    if role == "user":
        st.markdown(f'<div class="message-user">👤 <strong>You:</strong><br>{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-assistant">🤖 <strong>Assistant:</strong><br>{content}</div>', unsafe_allow_html=True)

        # Display sources if available
        if metadata:
            display_sources(metadata)


def display_sources(metadata):
    """
    Display source information with traceability

    Args:
        metadata: Dictionary containing route, sources, and other info
    """
    route = metadata.get('route', 'unknown')
    sources = metadata.get('sources', {})
    success = metadata.get('success', False)

    if not success:
        return

    with st.expander("📊 View Sources", expanded=False):
        if route == 'sql':
            # Display SQL query
            st.markdown("**🗄️ Data Source:** Database Query")
            sql_query = sources.get('sql_query', 'N/A')
            st.markdown(f'<div class="sql-query">{sql_query}</div>', unsafe_allow_html=True)

            row_count = sources.get('row_count', 0)
            st.caption(f"Retrieved {row_count} row(s) from database")

        elif route == 'rag':
            # Display document sources
            st.markdown("**📚 Data Source:** Documents")
            files = sources.get('files', [])

            if files:
                st.markdown("**Sources:**")
                for file_info in files:
                    display_name = file_info.get('display', file_info.get('file', 'Unknown'))
                    st.markdown(f"- 📄 {display_name}")
            else:
                st.caption("No specific sources cited")


def main():
    """Main application"""

    # Header
    st.markdown('<div class="main-header">💼 Financial Analyst Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about financial data (SQL) or company strategies (Documents)</div>', unsafe_allow_html=True)

    # Initialize chatbot
    chatbot = initialize_chatbot()

    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This hybrid chatbot answers questions using:

        **🗄️ SQL Database**
        - Company financials
        - Revenue, market cap, P/E ratios
        - Numerical comparisons

        **📚 Document Database**
        - Earnings transcripts
        - Strategic initiatives
        - AI plans, risks, outlook

        The system automatically routes your question to the appropriate data source!
        """)

        st.header("💡 Example Questions")

        with st.expander("💰 Financial Metrics (SQL)", expanded=False):
            st.markdown("""
            - What is Apple's revenue?
            - Which company has the highest market cap?
            - Compare revenue of Apple and Microsoft
            - Show me all technology companies
            - What is the average P/E ratio?
            """)

        with st.expander("📈 Strategic Insights (RAG)", expanded=False):
            st.markdown("""
            - What are Microsoft's AI initiatives?
            - What are NVIDIA's growth drivers?
            - What challenges does Apple face?
            - What is Meta's metaverse strategy?
            - What are the headwinds facing tech companies?
            """)

        st.markdown("---")

        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            chatbot.clear_conversation()
            st.reboot()

        st.markdown("---")
        st.caption("Built with ❤️ for financial analysis")

    # Display chat history
    for message in st.session_state.messages:
        display_message(
            message['role'],
            message['content'],
            message.get('metadata')
        )

    # Chat input
    if prompt := st.chat_input("Ask a question about companies or financials..."):
        # Display user message
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })
        display_message('user', prompt)

        # Get bot response
        with st.spinner("🤔 Thinking..."):
            try:
                result = chatbot.ask(prompt)

                answer = result['answer']
                metadata = {
                    'route': result['route'],
                    'sources': result['sources'],
                    'success': result['success']
                }

                # Display assistant message
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': answer,
                    'metadata': metadata
                })
                display_message('assistant', answer, metadata)

            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': error_msg
                })
                display_message('assistant', error_msg)
                st.error("Please check your API key and try again.")


if __name__ == "__main__":
    main()
