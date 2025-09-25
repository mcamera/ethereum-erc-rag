# Ethereum Request for Comments (ERCs) AI Agent

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for Ethereum Request for Comments (ERC) documents. It downloads and processes ERC markdown files from the [Ethereum ERC GitHub repository](https://github.com/ethereum/ERCs), extracting structured data that can be used for AI-powered analysis and question-answering about Ethereum improvement proposals.

⚠️ **This project is currently in development** and focuses on the initial data extraction phase of the RAG pipeline.

## Key Features

- 🔄 **Automated Repository Scraping**: Downloads and extracts markdown files from the Ethereum GitHub repository
- 📄 **Frontmatter Processing**: Parses YAML frontmatter from markdown files for structured metadata
- 🧹 **Content Cleaning**: Handles encoding issues and malformed files gracefully
- 📊 **Jupyter Notebook Integration**: Interactive exploration and analysis of ERC data
- 🛠️ **Modular Design**: Clean, reusable functions for repository data extraction

## Requirements

- Python 3.13+
- Dependencies managed with `uv` package manager

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

## Usage

### Running the Main Script

1. **Edit the main.py file** to specify your target repository:
   ```python
   repo_owner = "ethereum"  # Replace with target repo owner
   repo_name = "ERCs"       # Replace with target repo name
   ```

2. **Run the script**:
   ```bash
   python main.py
   ```

### Using the Jupyter Notebook

For interactive exploration and analysis:

1. **Start Jupyter**:
   ```bash
   uv run jupyter notebook
   ```

2. **Open** `read_repo_data.ipynb` and run the cells to:
   - Load and process ERC data
   - Explore document structure
   - Analyze metadata and content

## Output Format

The system returns a list of dictionaries, each containing:
- `content`: The markdown content of the file
- `metadata`: Parsed YAML frontmatter (if present)
- `filename`: Original file path in the repository

## Development

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **Jupyter** for interactive development

Run development tools:
```bash
# Format code
black .

# Sort imports
isort .
```

## Development Roadmap

### ✅ Ingest and Index the Data (Complete)
- [x] Fetch GitHub repository content
- [x] Extract documentation
- [x] Prepare data for search

### 🧠 Intelligent Processing for Data
- [ ] Cut and chunk the data for better search
- [ ] Split big documents using paragraphs and sections
- [ ] Apply intelligent chunking with AI

### 🔍 Add Search
- [ ] Build lexical search for exact matches and keywords
- [ ] Implement semantic search using embeddings
- [ ] Combine them with hybrid search

### 🤖 Agentic Search
- [ ] Implement RAG
- [ ] Defining tools for the agent
- [ ] Use Pydantic AI
- [ ] Make RAG agentic

### 🧪 Offline Evaluation and Testing
- [ ] Create evaluation datasets
- [ ] Evaluate search
- [ ] Use AI to evaluate the agent

### 🚀 Deploy the Agent
- [ ] Create UI for your agent with Streamlit
- [ ] Publish it on the Internet

## More info

This project follows the the AI Agents Course from Alexey Grigorev. Want to follow along?

👉 Sign up here: https://alexeygrigorev.com/aihero/

## License

See LICENSE file for details.
