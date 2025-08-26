# Iowa Wells RAG Chat Interface 🏔️

A beautiful and intelligent chat interface for exploring Iowa geological wells data using advanced RAG (Retrieval-Augmented Generation) technology.

## Features

- 🎯 **Intelligent Query Processing**: Ask natural language questions about Iowa wells
- 📊 **Context Source Display**: See exactly where information comes from
- 🗺️ **MCR Geographic Intelligence**: Understands Mid-Continental Rift vs non-MCR counties
- 💬 **Chat History**: Keep track of your conversation
- 🎨 **Professional Branding**: NHV logos and clean interface design
- ⚡ **Real-time Search**: Powered by Pinecone vector database and OpenAI embeddings
- 🔗 **Clickable Well Links**: Direct links to GeoSam well database

## Deployment

### Streamlit Cloud Deployment

1. **Environment Variables** (Set in Streamlit Cloud dashboard):
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

2. **Main File**: `iowa_wells_chat.py`

3. **Dependencies**: Automatically installed from `requirements.txt`

### Local Development

```bash
# Clone the repository
git clone https://github.com/LunarPerovskite/rag-test.git
cd rag-test

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (copy .env.example to .env and fill in values)
cp .env.example .env

# Run the application
streamlit run iowa_wells_chat.py
```

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Environment Variables**: 
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PINECONE_API_KEY`: Your Pinecone API key
3. **Vector Database**: The Pinecone index `nhv-iowa-wells` should be populated with 551 Iowa wells data

## MCR Geographic Intelligence

The system understands the Mid-Continental Rift (MCR) geography with 58 counties classified as within the rift zone. Ask questions like:
- "What's the deepest well inside the MCR?"
- "Compare formation depths between MCR and non-MCR counties"
- "Show me oil shows outside the MCR"

## Example Queries

Try asking questions like:

- "What are the deepest wells in Dallas County?"
- "Show me gas storage wells drilled by Sterling Drilling"
- "What is the typical bedrock depth in central Iowa counties?"
- "Find wells with total depth greater than 3000 feet"
- "Tell me about municipal wells in Polk County"
- "What drilling companies are most active in the database?"
- "Show wells drilled after 2020 with depth over 2500 feet"
- "Compare well depths between counties"

## Interface Features

### Main Chat Area
- Natural language conversation with the AI
- Real-time typing indicators
- Message history preservation
- Example query buttons for quick start

### Context Sources Panel
- **Source Ranking**: See which documents are most relevant
- **Well Metadata**: Complete information about each well
- **Relevance Scores**: Understand how well sources match your query
- **File References**: Know exactly which documents contain the information

### Sidebar Controls
- **Connection Status**: Real-time database connection monitoring
- **Database Statistics**: Live stats from your Pinecone index
- **Chat Settings**: Customize source display and limits
- **Clear History**: Start fresh conversations

## Technical Architecture

### Backend Components
- **LlamaIndex**: Advanced RAG framework for document processing
- **Pinecone**: Vector database for semantic search
- **OpenAI**: GPT-3.5-turbo for responses, text-embedding-3-large for embeddings
- **Streamlit**: Modern web interface framework

### Data Processing
- **Embedding Model**: `text-embedding-3-large` (3072 dimensions)
- **Chunk Strategy**: 3000 characters with 500 character overlap
- **Metadata Schema**: Comprehensive well identification and geographic information
- **Multi-format Support**: JSON metadata + PDF documents

### Performance Features
- **Similarity Search**: Top-K retrieval with configurable limits
- **Response Modes**: Compact responses for faster processing
- **Connection Pooling**: Efficient database connections
- **Error Handling**: Graceful fallbacks and user-friendly error messages

## Database Information

The chat interface connects to the `nhv-iowa-wells` Pinecone index containing:

- **551 Iowa Geological Wells** across 64 counties
- **6,266 Text Chunks** with comprehensive metadata
- **2,432 PDF Documents** with full-text search capability
- **Zero Processing Errors** - high-quality, clean data

### Metadata Fields Available

Each source includes:
- `well_id`: Unique well identifier
- `county`: Iowa county location
- `owner_name`: Well owner information
- `total_depth`: Well depth in feet
- `well_types`: Classification (municipal, gas storage, etc.)
- `drilling_company`: Company that drilled the well
- `content_type`: JSON metadata or PDF content
- `file_name`: Source document reference

## Troubleshooting

### Common Issues

1. **"API keys not found"**
   - Check that `.env` file exists in `../8_vectordb/`
   - Verify OPENAI_API_KEY and PINECONE_API_KEY are set

2. **"Connection failed"**
   - Ensure Pinecone index `nhv-iowa-wells` exists
   - Check internet connectivity
   - Verify API keys are valid

3. **"Import errors"**
   - Run `pip install -r requirements.txt`
   - Try upgrading: `pip install --upgrade -r requirements.txt`

4. **"Streamlit not found"**
   - Install manually: `pip install streamlit`
   - Use the launcher scripts which auto-install dependencies

### Performance Tips

1. **Adjust Source Count**: Use the sidebar slider to control how many sources are retrieved
2. **Clear History**: Reset the chat if responses become slow
3. **Specific Queries**: More specific questions get better, faster results
4. **Use Examples**: Click the example queries for optimal question formatting

## Files Structure

```
9_UI/
├── iowa_wells_chat.py      # Main Streamlit application
├── launch_chat.py          # Python launcher script
├── launch_chat.ps1         # PowerShell launcher script
├── requirements.txt        # Python dependencies
└── README.md              # This documentation
```

## Development

### Extending the Interface

The chat interface is built with modularity in mind:

- **Custom Styling**: Modify the CSS in `iowa_wells_chat.py`
- **New Features**: Add functions to the `IowaWellsChatInterface` class
- **Query Customization**: Adjust retrieval parameters in `initialize_rag_system()`
- **UI Components**: Add new Streamlit components in the `run()` method

### Configuration Options

Key parameters you can adjust:

```python
# In iowa_wells_chat.py
similarity_top_k=5          # Number of sources to retrieve
response_mode="compact"     # Response generation mode
temperature=0.1             # LLM creativity (0.0-1.0)
dimensions=3072            # Embedding dimensions
```

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify your environment setup against the prerequisites
3. Ensure the vector database was built successfully using `../8_vectordb/build_pinecone_vectordb.py`
4. Test individual components (API keys, Pinecone connection) separately

---

**Built with ❤️ for Iowa geological data exploration**
