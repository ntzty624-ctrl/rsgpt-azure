# Rail Standards GPT Azure Function

Azure Function for Rail Standards GPT - A RAG (Retrieval-Augmented Generation) system that provides intelligent document search and Q&A capabilities for rail standards documentation.

## Versions

Python Version: 3.11.x
Azure CLI Version: 4.2.1

## Environment Variables

The following environment variables need to be configured:

### Required Variables

| Variable | Description | Where to Find |
|----------|-------------|---------------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Azure Portal > Azure OpenAI Service > Keys and Endpoint |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint URL | Azure Portal > Azure OpenAI Service > Keys and Endpoint |
| `CHROMADB_HOST` | ChromaDB server hostname | Your ChromaDB deployment |
| `CHROMADB_PORT` | ChromaDB server port | Your ChromaDB deployment (default: 8000) |
| `CHROMADB_TOKEN` | ChromaDB authentication token | DEV KeyVault (for development) |

### Optional Variables

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `AZURE_EMBEDDING_DEPLOYMENT` | Azure OpenAI embedding model deployment name | `text-embedding-ada-002` |

### Configuration Details

- **Azure OpenAI Deployment**: `gpt-4o-mini` (hardcoded in config)
- **API Version**: `2024-12-01-preview` (hardcoded in config)
- **ChromaDB Collection**: `documents` (hardcoded in config)

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rs-gpt-azure-function
   ```

2. **Create environment file**
   ```bash
   cp .env.example .env
   ```

3. **Configure environment variables**  
   Edit `.env` file with your actual values (see Environment Variables section above)

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Deploy to Azure Functions**
   ```bash
   func azure functionapp publish <your-function-app-name>
   ```