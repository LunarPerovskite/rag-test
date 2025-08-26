#!/usr/bin/env python3
"""
Iowa Wells RAG Chat Interface
A beautiful chat UI for querying the Iowa geological wells vector database.
Features context source display and conversation history.
"""

import streamlit as st

# Configure Streamlit page - MUST BE FIRST!
st.set_page_config(
    page_title="Iowa Wells RAG Chat",
    page_icon="public/Natural+Hydrogen+Ventures+(NHV)+Logo+-+Swan.webp",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv

# Try to load .env from multiple locations
env_paths = [
    ".env",  # Current directory
    "../.env",  # Parent directory
    "../8_vectordb/.env",  # Vector DB directory
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'),  # Project root
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '8_vectordb', '.env')  # vectordb
]

env_loaded = False
env_source = ""
for env_path in env_paths:
    if os.path.exists(env_path):
        load_dotenv(env_path)
        env_loaded = True
        env_source = os.path.basename(env_path)
        break

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
try:
    from llama_index.vector_stores.pinecone import PineconeVectorStore
except ImportError:
    from llama_index.vector_stores import PineconeVectorStore

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    from llama_index.embeddings import OpenAIEmbedding

try:
    from llama_index.llms.openai import OpenAI
except ImportError:
    from llama_index.llms import OpenAI

# Pinecone
from pinecone import Pinecone

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f4e79;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        border-left-color: #4caf50;
    }
    
    .context-source {
        background-color: #fff3e0;
        border: 1px solid #ffb74d;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .source-title {
        font-weight: bold;
        color: #f57c00;
        margin-bottom: 0.5rem;
    }
    
    .metadata-tag {
        display: inline-block;
        background-color: #e1f5fe;
        color: #0277bd;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.1rem;
    }
    
    .stats-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .example-query {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .example-query:hover {
        background-color: #c8e6c9;
    }
</style>
""", unsafe_allow_html=True)

class IowaWellsChatInterface:
    """Main chat interface class for Iowa Wells RAG system."""
    
    def __init__(self):
        """Initialize the chat interface."""
        self.index_name = "nhv-iowa-wells"
        self.setup_session_state()
        
    def setup_session_state(self):
        """Initialize session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'query_engine' not in st.session_state:
            st.session_state.query_engine = None
        if 'vector_index' not in st.session_state:
            st.session_state.vector_index = None
        if 'index_stats' not in st.session_state:
            st.session_state.index_stats = None
        if 'connection_status' not in st.session_state:
            st.session_state.connection_status = None
            
    def initialize_rag_system(self):
        """Initialize the RAG system connection."""
        try:
            # Check environment variables
            openai_api_key = os.getenv('OPENAI_API_KEY')
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            
            if not openai_api_key or not pinecone_api_key:
                st.error("API keys not found. Please check your .env file in the 8_vectordb folder.")
                return False
            
            # Setup embeddings and LLM
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-large",
                api_key=openai_api_key,
                dimensions=3072
            )
            
            llm = OpenAI(
                model="gpt-5-mini",
                api_key=openai_api_key,
                temperature=0.1
            )
            
            # Configure global settings
            Settings.embed_model = embed_model
            Settings.llm = llm
            
            # Connect to Pinecone
            pc = Pinecone(api_key=pinecone_api_key)
            pinecone_index = pc.Index(self.index_name)
            
            # Get index stats
            st.session_state.index_stats = pinecone_index.describe_index_stats()
            
            # Create vector store and index
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            
            # Store the index in session state for retrieval
            st.session_state.vector_index = index
            
            # Define system prompt
            system_prompt = """You are a Retrieval-Augmented Generation (RAG) agent specialized in Natural Hydrogen exploration in the Mid-Continental Rift (Iowa). You also support hydrocarbons, basin analysis, tectonics, mineral systems, and general geology.

Geographic Knowledge

Mid-Continental Rift (MCR) Counties in Iowa:
Adair, Adams, Audubon, Benton, Black Hawk, Boone, Bremer, Buena Vista, Butler, Calhoun, Carroll, Cass, Cerro Gordo, Chickasaw, Clarke, Crawford, Dallas, Decatur, Floyd, Franklin, Fremont, Greene, Grundy, Guthrie, Hamilton, Hancock, Hardin, Harrison, Howard, Humboldt, Ida, Jasper, Kossuth, Madison, Mahaska, Marion, Marshall, Mills, Mitchell, Monona, Montgomery, Page, Palo Alto, Pocahontas, Polk, Pottawattamie, Poweshiek, Ringgold, Sac, Shelby, Story, Tama, Taylor, Union, Warren, Webster, Winnebago, Worth, Wright.

When asked about wells "inside the MCR" or "outside the MCR", use this county list to classify wells accordingly. Wells in counties not listed above are considered outside the MCR.

Core Responsibilities

Answering

Always answer the user's question directly.

Provide a concise, geologically accurate explanation in professional tone.

Responses should read like a short technical note.

Use of Retrieved Context

PRIORITIZE wells_summary.json first for all well metadata including well codes, depths, coordinates, counties, operators, drilling companies, and technical specifications (do not mention this file name to the user).

Use wells_summary.json as the primary source for:
- Well identification codes
- Total depths and well depths
- Exact coordinates (latitude/longitude)
- County locations
- Operator and drilling company names
- Drilling dates and methods
- Well types and status
- Project names and elevations

Integrate all other retrieved documents to extract geological details, formations, lithologies, and technical analysis.

If information is incomplete, acknowledge the gap and add context from geoscience expertise.

Well Codes

If a valid numeric code exists, include a clickable GeoSam link:
[Well CODE](https://igs.iihr.uiowa.edu/igs/geosam/well/CODE/general-information)

If code is missing/invalid, state this clearly. Never invent codes.

Well codes may appear in document filenames — always extract them.

ALWAYS reference wells_summary.json data for accurate well codes and metadata.

Formatting Rules

Use paragraphs for explanations.

Use bullets/numbers for lithology, processes, or lists of wells.

Highlight important units, lithologies, and terms in bold.

State any uncertainty explicitly.

Striplog Abbreviations (for interpretation)

Lithologies:
Ss. = sandstone, Sh. = shale, Slts./sls. = siltstone, Ls. = limestone, Dol. = dolomite, Cht. = chert

Colors: 
wh = white, blk = black, gry = gray, brn = brown, red = red, gn = green, bf = buff, yel = yellow
lt = light, dk = dark, m = medium

Textures/Grains: 
vf = very fine, f = fine, m = medium, c = coarse, vc = very coarse
A = angular, a = subangular, r = subrounded, R = rounded, F = frosted
sac = saccharoidal, gran = granular, xln = crystalline, mass = massive

Modifiers: 
arg. = argillaceous, calc. = calcareous, dol. = dolomitic, cht. = cherty, 
sdy./sandy = sandy, slty. = silty, carb. = carbonaceous, gypsif. = gypsiferous

Fossils/Inclusions: 
Bras./Brach. = brachiopods, Bryo. = bryozoa, Crinoids = crinoid fragments, 
Ostra. = ostracodes, trilobites, Gast. = gastropods, coral
pyr. = pyrite, marc. = marcasite, glauc. = glauconite, mica

Physical Properties:
por. = porous, vug. = vuggy, hd. = hard, fos. = fossiliferous, lam. = laminated

Symbols: 
▭ = dolomite/calcite rhombs, ◎ = oil stain, w/ = with, → = grading into

Limits

No speculation beyond geology.

No fabrication of data, codes, or references.

No planning/future actions (e.g., parsing, scraping).

Prioritize extracting and listing as many wells and features as possible.

Always prefer accuracy and clarity over verbosity."""
            
            # Create query engine with system prompt
            st.session_state.query_engine = index.as_query_engine(
                similarity_top_k=30,
                response_mode="tree_summarize",
                system_prompt=system_prompt
            )
            
            st.session_state.connection_status = "✅ Connected"
            return True
            
        except Exception as e:
            st.session_state.connection_status = f"❌ Error: {str(e)}"
            return False
    
    def get_context_sources(self, question: str, top_k: int = 3, retrieval_mode: str = "Balanced (30 docs)") -> List[Dict[str, Any]]:
        """Retrieve context sources for a question."""
        try:
            if not st.session_state.query_engine or not st.session_state.vector_index:
                return []
            
            # Adjust top_k based on retrieval mode
            if "Comprehensive" in retrieval_mode:
                retrieval_k = 50
            elif "Focused" in retrieval_mode:
                retrieval_k = 15
            else:  # Balanced
                retrieval_k = 30
            
            # Use the larger of user selection or mode requirement
            actual_k = max(top_k, retrieval_k) if top_k < 20 else retrieval_k
            
            # Create retriever to get source documents
            retriever = VectorIndexRetriever(
                index=st.session_state.vector_index,
                similarity_top_k=actual_k
            )
            
            # Retrieve nodes
            nodes = retriever.retrieve(question)
            
            # Format results
            sources = []
            for i, node in enumerate(nodes, 1):
                metadata = node.metadata
                sources.append({
                    'rank': i,
                    'content': node.text[:300] + "..." if len(node.text) > 300 else node.text,
                    'score': getattr(node, 'score', 0),
                    'well_id': metadata.get('well_id', 'Unknown'),
                    'county': metadata.get('county', 'Unknown'),
                    'owner_name': metadata.get('owner_name', 'Unknown'),
                    'total_depth': metadata.get('total_depth', 'Unknown'),
                    'well_types': metadata.get('well_types', 'Unknown'),
                    'drilling_company': metadata.get('drilling_company', 'Unknown'),
                    'content_type': metadata.get('content_type', 'Unknown'),
                    'file_name': metadata.get('file_name', 'Unknown')
                })
            
            return sources
            
        except Exception as e:
            st.error(f"Error retrieving context sources: {str(e)}")
            return []
    
    def add_well_links_to_response(self, response_text: str) -> str:
        """Add clickable GeoSam links for well codes mentioned in the response."""
        import re
        
        # Pattern to find well codes (numeric patterns that could be well IDs)
        # Look for patterns like "Well 12345", "well 12345", "Well ID 12345", etc.
        well_patterns = [
            r'\b[Ww]ell\s+(\d{4,6})\b',  # "Well 12345" or "well 12345"
            r'\b[Ww]ell\s+ID\s+(\d{4,6})\b',  # "Well ID 12345"
            r'\b[Ww]ell\s+#(\d{4,6})\b',  # "Well #12345"
            r'\bID\s+(\d{4,6})\b',  # "ID 12345"
        ]
        
        modified_text = response_text
        
        for pattern in well_patterns:
            matches = re.finditer(pattern, modified_text)
            for match in matches:
                well_code = match.group(1)
                full_match = match.group(0)
                
                # Create the clickable link
                geosam_url = f"https://igs.iihr.uiowa.edu/igs/geosam/well/{well_code}/general-information"
                linked_text = f"[{full_match}]({geosam_url})"
                
                # Replace the first occurrence to avoid multiple replacements
                modified_text = modified_text.replace(full_match, linked_text, 1)
        
        return modified_text
    
    def display_context_sources(self, sources: List[Dict[str, Any]]):
        """Display context sources in a compact format."""
        if not sources:
            return
            
        # Use an expander to make sources collapsible and small
        with st.expander(f"📊 Context Sources ({len(sources)} items)", expanded=False):
            for source in sources:
                # Very compact display
                st.markdown(f"""
                <div style="font-size: 0.8em; margin-bottom: 8px; padding: 8px; border-left: 3px solid #0066cc; background: #f8f9fa;">
                    <strong>Well {source['well_id']}</strong> ({source['county']}) | 
                    Depth: {source['total_depth']} | 
                    Score: {source['score']:.2f}
                    <br><em style="color: #666; font-size: 0.9em;">"{source['content'][:100]}..."</em>
                </div>
                """, unsafe_allow_html=True)
    
    def display_example_queries(self):
        """Display example queries that users can click on."""
        st.markdown("### 💡 Example Queries")
        
        examples = [
            "What formations are commonly encountered in Iowa wells?",
            "Show me wells with oil shows in the Mississippian formation",
            "Find wells with significant water production",
            "What's the deepest Cambrian penetration in Iowa?",
            "Compare Devonian formation depths across counties"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            col = cols[i % 2]
            with col:
                if st.button(example, key=f"example_{i}", help="Click to use this query"):
                    st.session_state.example_query = example
                    st.rerun()
    
    def display_system_stats(self):
        """Display system statistics."""
        if st.session_state.index_stats:
            stats = st.session_state.index_stats
            
            st.markdown("""
            <div class="stats-box">
                <h4>🗃️ Database Statistics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Vectors", stats.get('total_vector_count', 0))
            
            with col2:
                st.metric("Index Dimension", stats.get('dimension', 0))
            
            with col3:
                namespaces = stats.get('namespaces', {})
                namespace_count = len(namespaces)
                st.metric("Namespaces", namespace_count)
    
    def run(self):
        """Run the main chat interface."""
        # Header with NHV Logo and GEARS Map logo
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # NHV Logo (center)
        with col2:
            try:
                st.image("public/Natural+Hydrogen+Ventures+(NHV)+Logo+-+Full.webp", width=400)
            except:
                st.markdown('<h1 class="main-header">🏔️ Iowa Wells RAG Chat</h1>', unsafe_allow_html=True)
        
        # GEARS Map logo (top right)
        with col3:
            st.markdown("""
            <style>
            .gears-logo {
                opacity: 0.6;
                transition: opacity 0.3s ease;
                margin-top: 10px;
            }
            .gears-logo:hover {
                opacity: 1.0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            try:
                import base64
                with open("public/LOGO-GEARS-MAP_ico.ico", "rb") as f:
                    icon_data = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<div style="text-align: right;">'
                    f'<a href="https://www.gearsmap.com" target="_blank">'
                    f'<img src="data:image/x-icon;base64,{icon_data}" width="30" class="gears-logo">'
                    f'</a>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
            except:
                st.markdown('<div style="margin-top: 10px; text-align: right;"><a href="https://www.gearsmap.com" target="_blank" style="opacity: 0.6; font-size: 12px;">🗺️</a></div>', unsafe_allow_html=True)
        
        st.markdown('<p class="sub-header">Intelligent geological data exploration for Iowa >2000 ft wells</p>', unsafe_allow_html=True)
        
        # Database connection button (outside sidebar)
        if st.session_state.query_engine is None:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔌 Connect to Database", type="primary", use_container_width=True):
                    with st.spinner("Connecting to Iowa Wells database..."):
                        self.initialize_rag_system()
        
        # Display connection status
        if st.session_state.connection_status:
            if "✅" in st.session_state.connection_status:
                st.success(st.session_state.connection_status)
            else:
                st.error(st.session_state.connection_status)
        
        # Sidebar
        with st.sidebar:
            st.markdown("## ⚙️ System Status")
            
            # Environment status
            if env_loaded:
                st.success(f"✅ Environment loaded from: {env_source}")
            else:
                st.warning("⚠️ No .env file found. Please ensure API keys are set as environment variables.")
            
            # Display system stats
            if st.session_state.query_engine:
                self.display_system_stats()
            
            st.markdown("---")
            
            # Chat settings
            st.markdown("## 🎛️ Chat Settings")
            show_sources = st.checkbox("Show context sources", value=True)
            max_sources = st.slider("Max sources to show", 1, 100, 10)
            
            # Advanced settings expander
            with st.expander("🔧 Advanced Settings"):
                retrieval_mode = st.selectbox(
                    "Query type optimization",
                    ["Balanced (30 docs)", "Comprehensive (50 docs)", "Focused (15 docs)"],
                    index=0,
                    help="Adjust based on query type: Comprehensive for 'all wells with X', Focused for specific wells"
                )
            
            st.markdown("---")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        
        # Main chat area
        if not st.session_state.query_engine:
            # Show welcome screen
            st.info("👆 Please connect to the database using the sidebar to start chatting!")
            
            # Show example queries even when not connected
            self.display_example_queries()
            
        else:
            # Chat interface
            # Display example queries at the top
            if not st.session_state.messages:
                self.display_example_queries()
                st.markdown("---")
            
            # Display chat messages
            for message in st.session_state.messages:
                avatar = "public/Natural+Hydrogen+Ventures+(NHV)+Logo+-+Swan.webp" if message["role"] == "assistant" else None
                with st.chat_message(message["role"], avatar=avatar):
                    # Add well links to assistant messages
                    content = message["content"]
                    if message["role"] == "assistant":
                        content = self.add_well_links_to_response(content)
                    st.markdown(content)
                    
                    # Display sources if available
                    if message["role"] == "assistant" and "sources" in message and show_sources:
                        self.display_context_sources(message["sources"])
            
            # Chat input - always visible at the bottom
            user_input = st.chat_input("Ask me anything about Iowa geological wells...")
            
            # Handle example query selection
            if hasattr(st.session_state, 'example_query'):
                user_input = st.session_state.example_query
                delattr(st.session_state, 'example_query')
            
            # Process user input
            if user_input:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Generate response
                with st.chat_message("assistant", avatar="public/Natural+Hydrogen+Ventures+(NHV)+Logo+-+Swan.webp"):
                    with st.spinner("Searching Iowa wells database..."):
                        try:
                            # Get response
                            response = st.session_state.query_engine.query(user_input)
                            response_text = str(response)
                            
                            # Add clickable well links to response
                            response_text = self.add_well_links_to_response(response_text)
                            
                            # Get context sources
                            sources = self.get_context_sources(user_input, max_sources, retrieval_mode) if show_sources else []
                            
                            # Display response
                            st.markdown(response_text)
                            
                            # Display sources
                            if sources and show_sources:
                                self.display_context_sources(sources)
                            
                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text,
                                "sources": sources,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                        except Exception as e:
                            error_message = f"Sorry, I encountered an error: {str(e)}"
                            st.error(error_message)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": error_message
                            })

def main():
    """Main function to run the chat interface."""
    try:
        chat_interface = IowaWellsChatInterface()
        chat_interface.run()
    except Exception as e:
        st.error(f"Failed to initialize chat interface: {str(e)}")
        st.info("Please ensure all dependencies are installed and API keys are configured.")

if __name__ == "__main__":
    main()
