# Ethereum Request for Comments (ERCs) AI Agent

## Project Overview

This project implements a comprehensive Retrieval-Augmented Generation (RAG) system for Ethereum Request for Comments (ERC) documents. It downloads and processes ERC markdown files from the [Ethereum ERC GitHub repository](https://github.com/ethereum/ERCs), extracting structured data that can be used for AI-powered analysis and question-answering about Ethereum improvement proposals.

⚠️ **This project is currently in development**

## Key Features

- 🔄 **Automated Repository Scraping**: Downloads and extracts markdown files from the Ethereum GitHub repository using `read_repo_data`
- 📄 **Frontmatter Processing**: Parses YAML frontmatter from markdown files for structured metadata
- 🧠 **Intelligent Document Processing**: Multiple chunking strategies including sliding window, section-based splitting, and AI-powered intelligent chunking
- 🔍 **Hybrid Search Engine**: Combines lexical search (MinSearch) with vector search (sentence transformers) for optimal retrieval
- 🤖 **Pydantic AI Agent**: Implements an intelligent agent with tool calling capabilities for ERC Q&A
- 📊 **Comprehensive Evaluation**: LLM-as-a-judge evaluation system with structured assessment criteria
- 🛠️ **Interactive Development**: Full Jupyter notebook implementation with detailed analysis and experimentation

## Requirements

- Python 3.13+
- Dependencies managed with `uv` package manager
- Google AI API key (for Gemini models)

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mcamera/ethereum-erc-rag.git
   cd ethereum-erc-rag
   ```

2. **Install dependencies** (using uv):
   ```bash
   uv sync --dev
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

## Usage

### Running the Main Script

1. **Configure the system** in `main.py`:
   ```python
   repo_owner = "ethereum"
   repo_name = "ERCs"
   chunking_method = "sliding_window"  # Options: sliding_window, split_markdown_by_level, intelligent_chunking
   search_method = "text"  # Options: text, vector, hybrid
   query = "What is ERC-4337?"  # Your question
   ```

2. **Run the conversational agent**:
   ```bash
   python main.py
   ```

### Using the Interactive Notebook

For detailed exploration and development:

1. **Start Jupyter**:
   ```bash
   uv run jupyter notebook
   ```

2. **Open** `ai_agent_notebook.ipynb` and explore:
   - Document ingestion and processing pipeline
   - Comparison of chunking strategies (sliding window vs. section-based vs. AI-powered)
   - Search implementation (lexical, vector, hybrid)
   - Agent development with Pydantic AI
   - Comprehensive evaluation and testing framework

## Architecture

### Document Processing Pipeline
- **Data Ingestion**: `read_repo_data()` downloads and processes GitHub repository content
- **Chunking Strategies**:
  - **Sliding Window**: `sliding_window()` with configurable size (2000) and step (1000)
  - **Section-Based**: `split_markdown_by_level()` splits by markdown headers
  - **AI-Powered**: `intelligent_chunking()` uses Gemini for semantic chunking

### Search Implementation
- **Lexical Search**: MinSearch index with TF-IDF scoring on multiple text fields
- **Vector Search**: Sentence transformers (`multi-qa-distilbert-cos-v1`) with semantic similarity
- **Hybrid Search**: Combines both approaches with deduplication for optimal results

### Conversational AI Agent
- **Framework**: Pydantic AI with Google Gemini 2.5 Flash models
- **Tools**: `text_search()` function integrated as agent tool for document retrieval
- **Capabilities**: Intelligent question answering with automatic source citations
- **References**: Automatic GitHub link generation for source materials

### Evaluation Framework
- **Structured Logging**: Comprehensive interaction logging with JSON serialization
- **LLM-as-Judge**: Automated evaluation using structured criteria:
  - Instructions adherence and avoidance patterns
  - Answer relevance, clarity, and completeness
  - Citation accuracy and tool usage verification
- **Test Generation**: AI-powered question generation from real ERC content
- **Quantitative Analysis**: Pandas-based statistical evaluation of agent performance

## Output Format

The system processes ERC documents into structured format with:
- `content`: The markdown content of the file
- `metadata`: Parsed YAML frontmatter (title, author, status, type, etc.)
- `filename`: Original file path in the repository
- `chunk`/`section`: Processed content segments optimized for search
- `start`: Position indicators for sliding window chunks

## Development

This project uses modern Python development practices:
- **UV** for fast dependency management
- **Black** for code formatting
- **isort** for import sorting  
- **Jupyter** for interactive development and experimentation
- **Pydantic** for type safety and validation

Run development tools:
```bash
# Format code
black .

# Sort imports
isort .

# Sync dependencies
uv sync --dev
```

## Development Roadmap

### ✅ Ingest and Index the Data (Complete)
- [x] Fetch GitHub repository content
- [x] Extract and parse markdown documentation
- [x] Prepare data for search

### ✅ Intelligent Processing for Data (Complete)
- [x] Cut and chunk the data for better search
- [x] Split big documents using paragraphs and sections
- [x] Apply intelligent chunking with AI

### ✅ Add Search (Complete)
- [x] Lexical search with MinSearch for keyword matching
- [x] Semantic search with sentence transformers
- [x] Hybrid search combining both approaches with result deduplication

### ✅ Agentic Search (Complete)
- [x] Implement RAG with Pydantic AI Agent
- [x] Define search tools for the agent
- [x] Create intelligent ERC documentation assistant with citation support
- [x] Build conversational interface with structured responses

### ✅ Offline Evaluation and Testing (Complete)
- [x] Comprehensive logging system
- [x] LLM-as-a-judge evaluation
- [x] Automated question generation for systematic testing
- [x] Multi-criteria evaluation framework with quantitative metrics
- [x] Pandas-based analysis of agent performance across test cases

### 🚀 Deploy the Agent (Next Phase)
- [ ] Create Streamlit UI for public access
- [ ] Deploy to cloud platform for live usage

## Evaluation Results

The system demonstrates strong performance across evaluation criteria:
- **Instructions Follow**: High adherence to system prompts and user requirements
- **Answer Relevance**: Consistently addresses user questions directly
- **Answer Clarity**: Provides clear, technically accurate responses
- **Citation Quality**: Automatically includes proper source references with GitHub links
- **Completeness**: Comprehensive coverage of technical topics
- **Tool Usage**: Effective search tool invocation for information retrieval

Performance metrics are tracked quantitatively using the pandas-based evaluation framework.

## Technical Stack

- **Python 3.13** with modern async/await patterns
- **Pydantic AI** for agent framework and structured outputs
- **Google Gemini 2.5** for LLM capabilities  
- **Sentence Transformers** for semantic embeddings
- **MinSearch** for efficient text and vector search
- **Jupyter** for interactive development
- **UV** for dependency management

## More info

This project follows the AI Agents Course from Alexey Grigorev. Want to follow along?

👉 Sign up here: https://alexeygrigorev.com/aihero/

## License

See LICENSE file for details.