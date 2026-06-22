## AI Document Assistant

AI Document Assistant is a secure multi-user Retrieval-Augmented Generation (RAG) chatbot that enables users to upload PDF documents and receive context-aware answers through natural language queries. The system combines semantic search using ChromaDB, keyword retrieval using BM25, and local Large Language Models (Llama 3 via Ollama) to generate accurate responses grounded in uploaded documents.

In addition to document intelligence, the platform provides user authentication, session management, chat history storage, document management, and real-time response streaming, creating a complete AI-powered document assistant experience.


## Architecture

```text
Document Processing Pipeline

Upload PDF
 ├─ PDF Text Extraction
 ├─ Content-aware Chunking
 ├─ Embedding Generation
 ├─ ChromaDB Vector Storage
 └─ BM25 Index Creation


Question Answering Pipeline

User Question
 ├─ Query Embedding
 ├─ Semantic Search (ChromaDB)
 ├─ Keyword Search (BM25)
 ├─ Hybrid Retrieval
 ├─ Context Construction
 ├─ Llama 3 (Ollama)
 └─ Response Generation


User Management Pipeline

User Registration
 ├─ Account Creation
 ├─ User Authentication
 ├─ Session Management
 ├─ Chat History Storage
 ├─ Document Tracking
 └─ Secure Logout
```

## Stack

| Layer | Technology |
|---------|------------|
| Backend | Flask (Python 3.11) |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| ORM | SQLAlchemy |
| Vector Store | ChromaDB |
| Lexical Search | BM25 |
| Embeddings | Ollama Embeddings |
| LLM | Llama 3 (Ollama) |
| PDF Processing | PyMuPDF |
| Authentication | Flask Session |
| Chat Storage | SQLite |
| Streaming | Flask Streaming Response |


## Chunking Strategy

The system uses recursive text chunking to preserve semantic meaning during document processing.

Chunking order:

Paragraphs → Lines → Sentences → Words

Each chunk contains:

- Filename
- Page Number
- Section Name
- Chunk Index

Default Configuration:

- Chunk Size: 500
- Chunk Overlap: 50

This approach helps maintain context while improving retrieval accuracy.


## Retrieval Pipeline

```text
The system follows a hybrid retrieval architecture.

User Question
│
├── Query Embedding Generation
│
├── Semantic Search (ChromaDB)
│
├── Keyword Search (BM25)
│
├── Reciprocal Rank Fusion (RRF)
│
├── Context Construction
│
└── LLM Response Generation
```


## Caching

The chatbot uses a persistent question cache to improve response speed.

Cache Features:

- SHA-256 Question Hashing
- Persistent JSON Storage
- Faster Repeated Queries
- Reduced LLM Processing

Benefits:

- Lower response time
- Reduced computation cost
- Improved user experience


## Document Processing Pipeline


```text
Upload PDF
│
├── SHA-256 Hash Generation
│
├── Duplicate Detection
│
├── PDF Text Extraction
│
├── Recursive Chunking
│
├── Embedding Generation
│
├── ChromaDB Storage
│
└── BM25 Index Creation
```


## Reranking

The system uses Cohere Rerank v3.5 Cross Encoder to improve retrieval quality.

Features:

- Query-aware ranking
- Semantic relevance scoring
- Noise reduction
- Improved answer accuracy


## Vector Database

The system stores document embeddings inside ChromaDB.

Stored Metadata:

- Filename
- Page Number
- Section Name
- Chunk Index
- Content Hash


