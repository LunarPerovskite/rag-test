#!/usr/bin/env python3
"""
Iowa Wells RAG Chat Application
A simple chat interface for querying the Iowa geological wells vector database.
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to path to import from 8_vectordb
sys.path.append(str(Path(__file__).parent.parent / "8_vectordb"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / "8_vectordb" / ".env")

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Pinecone
from pinecone import Pinecone

class IowaWellsChat:
    """Simple chat interface for Iowa Wells RAG system."""
    
    def __init__(self):
        self.index_name = "nhv-iowa-wells"
        self.setup_components()
        self.connect_to_index()
    
    def setup_components(self):
        """Setup LlamaIndex components."""
        # Get API keys
        openai_api_key = os.getenv('OPENAI_API_KEY')
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        
        if not openai_api_key or not pinecone_api_key:
            st.error("❌ API keys not found! Make sure OPENAI_API_KEY and PINECONE_API_KEY are set in your .env file.")
            st.stop()
        
        # Setup embeddings and LLM
        self.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=openai_api_key,
            dimensions=3072
        )
        
        self.llm = OpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.1
        )
        
        # Configure global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
    
    def connect_to_index(self):
        """Connect to the existing Pinecone index."""
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            
            # Connect to existing index
            self.pinecone_index = pc.Index(self.index_name)
            
            # Create vector store
            self.vector_store = PineconeVectorStore(
                pinecone_index=self.pinecone_index
            )
            
            # Create index from existing vector store
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error connecting to Pinecone index: {str(e)}")
            return False
    
    def query(self, question: str, top_k: int = 5):
        """Query the vector database and return response with context."""
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="compact"
            )
            
            # Execute query
            response = query_engine.query(question)
            
            # Get context sources
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k
            )
            
            nodes = retriever.retrieve(question)
            
            # Format context information
            context_info = []
            for i, node in enumerate(nodes, 1):
                metadata = node.metadata
                well_id = metadata.get('well_id', 'Unknown')
                county = metadata.get('county', 'Unknown')
                content_type = metadata.get('content_type', 'unknown')
                file_name = metadata.get('file_name', 'unknown')
                score = getattr(node, 'score', 0)
                
                context_info.append({
                    'rank': i,
                    'well_id': well_id,
                    'county': county,
                    'content_type': content_type,
                    'file_name': file_name,
                    'score': score,
                    'preview': node.text[:200] + "..." if len(node.text) > 200 else node.text
                })
            
            return str(response), context_info
            
        except Exception as e:
            return f"❌ Error: {str(e)}", []

def main():
    """Main Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="Iowa Wells RAG Chat",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
        background-color: #f8f9fa;
    }
    .context-source {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #2a5298;
    }
    .source-metadata {
        font-size: 0.8rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Iowa Wells RAG Chat</h1>
        <p>Ask questions about Iowa geological wells data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat system
    if 'chat_system' not in st.session_state:
        with st.spinner("🔌 Connecting to Iowa Wells database..."):
            st.session_state.chat_system = IowaWellsChat()
        st.success("✅ Connected to Iowa Wells vector database!")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar with example queries
    with st.sidebar:
        st.header("📝 Example Queries")
        
        example_queries = [
            "What are the deepest wells in Dallas County?",
            "Show me wells drilled by Sterling Drilling",
            "Tell me about gas storage wells in Iowa",
            "What is the typical bedrock depth in central Iowa?",
            "Find wells with total depth greater than 3000 feet",
            "Show wells owned by Northern Natural Gas",
            "What drilling companies work in Polk County?",
            "Tell me about wells drilled after 2020"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query[:20]}"):
                st.session_state.current_query = query
        
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("- Ask about specific counties, companies, or well types")
        st.markdown("- Inquire about depths, drilling dates, or locations")
        st.markdown("- Request comparisons between different areas")
        
        st.markdown("---")
        st.markdown("**📊 Database Info:**")
        st.markdown("- 551 wells processed")
        st.markdown("- 64 counties covered")
        st.markdown("- 6,266 text chunks")
        st.markdown("- 2,432 PDF documents")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show context sources for assistant messages
                if message["role"] == "assistant" and "context" in message:
                    with st.expander("📍 Sources Used", expanded=False):
                        for source in message["context"]:
                            st.markdown(f"""
                            <div class="context-source">
                                <div class="source-metadata">
                                    <strong>Well {source['well_id']}</strong> ({source['county']} County) | 
                                    {source['content_type']} | Score: {source['score']:.3f}
                                </div>
                                <small>{source['preview']}</small>
                            </div>
                            """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask about Iowa wells..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response from RAG system
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching Iowa wells database..."):
                    response, context = st.session_state.chat_system.query(prompt, top_k=5)
                
                st.markdown(response)
                
                # Show context sources
                if context:
                    with st.expander("📍 Sources Used", expanded=False):
                        for source in context:
                            st.markdown(f"""
                            <div class="context-source">
                                <div class="source-metadata">
                                    <strong>Well {source['well_id']}</strong> ({source['county']} County) | 
                                    {source['content_type']} | Score: {source['score']:.3f}
                                </div>
                                <small>{source['preview']}</small>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "context": context
            })
        
        # Handle example query button clicks
        if hasattr(st.session_state, 'current_query'):
            prompt = st.session_state.current_query
            del st.session_state.current_query
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get response from RAG system
            with st.spinner("🔍 Searching Iowa wells database..."):
                response, context = st.session_state.chat_system.query(prompt, top_k=5)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "context": context
            })
            
            st.rerun()
    
    with col2:
        st.subheader("ℹ️ About")
        
        st.markdown("""
        **Iowa Wells RAG System**
        
        This chat interface allows you to query a comprehensive database of Iowa geological wells using natural language.
        
        **Data Sources:**
        - Iowa Geological Survey wells
        - Well logs and drilling records
        - Geophysical measurements
        - Construction details
        
        **Capabilities:**
        - Search by location (county, coordinates)
        - Filter by depth, drilling company, dates
        - Find specific well types (gas storage, water, etc.)
        - Compare wells across regions
        
        **Technology:**
        - Pinecone vector database
        - OpenAI GPT-3.5 Turbo
        - Text-embedding-3-large
        - LlamaIndex RAG framework
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
