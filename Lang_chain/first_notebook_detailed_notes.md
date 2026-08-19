# 📒 Detailed Notes: `first.ipynb` — LangChain Training Notebook

> A line-by-line explanation of every cell in your notebook, covering **why** each step is used, what it does, and **alternative methods** you could use instead.

---

## Cell 1 — Markdown: Introduction

```
### Getting Started with Langchain and OPEN AI
Setup LangSmith and LangServe
LangChain: prompt templates, models, and output parsers.
build simple application with langchain
trace your application with langsmith
server your applicaiton with langserve
```

> [!NOTE]
> This cell outlines 3 pillars of LangChain ecosystem:
> - **LangChain** → The core framework (prompt templates, models, output parsers, chains)
> - **LangSmith** → Observability & tracing platform (monitors every LLM call, token usage, latency)
> - **LangServe** → Deploys your chain/agent as a REST API with one command

---

## Cell 2 — Agent with `create_agent` (Shorthand Method)

```python
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What are you made of?"}]}
)
print(response)
print("\nFinal Answer:", response["messages"][-1].content)
```

### Step-by-step Explanation

| Line | What it does | Why |
|------|-------------|-----|
| `from dotenv import load_dotenv` | Imports the `load_dotenv` function from the `python-dotenv` package | Needed to read `.env` file |
| `load_dotenv()` | Reads `.env` file and injects `OPENAI_API_KEY`, `LANGCHAIN_API_KEY` etc. into `os.environ` | LangChain auto-reads these environment variables — you don't need to pass keys manually |
| `from langchain.agents import create_agent` | Imports the high-level agent factory function | `create_agent` is the simplest way to build a ReAct agent using LangGraph under the hood |
| `from langchain_core.tools import tool` | Imports the `@tool` decorator | Converts any Python function into a LangChain-compatible tool |
| `@tool` decorator | Wraps `get_weather` so LangChain knows its **name**, **description** (from the docstring), and **input schema** (from the type hints) | The LLM reads the tool's description to decide **when** to call it |
| `model="openai:gpt-4o-mini"` | Shorthand string format — LangChain parses the provider (`openai`) and model name (`gpt-4o-mini`) automatically | No need to import `ChatOpenAI` separately |
| `agent.invoke(...)` | Runs the agent's full ReAct loop (Reason → Act → Observe → repeat) | Returns a dict with `"messages"` containing the full conversation |
| `response["messages"][-1].content` | Gets the **last message** (the final AI answer) from the conversation | The agent may call tools multiple times; the last message is always the final answer |

### 🔄 Alternative Methods

```python
# ---- ALT 1: Using OpenAI SDK directly (no LangChain) ----
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env automatically
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What are you made of?"}]
)
print(response.choices[0].message.content)
# ⚠️ Limitation: No agent loop, no tool calling, no memory

# ---- ALT 2: Using LangGraph directly (lower-level) ----
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools=[get_weather])
# This gives more control over the agent's state graph

# ---- ALT 3: Using a different model provider ----
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
# Rest of the code stays the same!
```

---

## Cell 3 — Agent with Explicit LLM Initialization

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = ChatOpenAI(model="gpt-4o-mini")

system_message = "You are a helpful assistant."
agent = create_agent(llm, tools=[get_weather], system_prompt=system_message)

response = agent.invoke({"messages": [("user", "What is the weather in San Francisco?")]})
print("\nFinal Answer:", response["messages"][-1].content)
```

### What's Different from Cell 2?

| Aspect | Cell 2 (Shorthand) | Cell 3 (Explicit) |
|--------|--------------------|--------------------|
| Model init | `model="openai:gpt-4o-mini"` (string) | `ChatOpenAI(model="gpt-4o-mini")` (object) |
| Flexibility | Less — can't customize temperature, max_tokens etc. | More — can pass `temperature=0.7`, `max_tokens=500` etc. |
| Message format | `{"role": "user", "content": "..."}` (dict) | `("user", "...")` (tuple shorthand) |

> [!TIP]
> **Use the explicit method (Cell 3) when you need fine-grained control:**
> ```python
> llm = ChatOpenAI(
>     model="gpt-4o-mini",
>     temperature=0,        # Deterministic output
>     max_tokens=500,       # Limit response length
>     request_timeout=30,   # Timeout in seconds
> )
> ```

### How the Agent Loop Works Internally

```
User: "What is the weather in San Francisco?"
         ↓
   LLM THINKS: "I should use the get_weather tool"
         ↓
   CALLS TOOL: get_weather("San Francisco")
         ↓
   TOOL RETURNS: "It's always sunny in San Francisco!"
         ↓
   LLM THINKS: "Now I have the answer"
         ↓
   FINAL ANSWER: "The weather in San Francisco is always sunny!"
```

---

## Cell 4 — RAG: Loading a Text File

```python
from langchain_community.document_loaders import TextLoader

docs = TextLoader('speech.txt')

print(docs)
print(docs.load())
```

### Step-by-step Explanation

| Line | What it does | Why |
|------|-------------|-----|
| `TextLoader('speech.txt')` | Creates a loader object (does NOT read the file yet) | LangChain uses "lazy loading" — it only reads when you call `.load()` |
| `docs.load()` | Actually reads the file and returns a `list[Document]` | Each `Document` has `.page_content` (the text) and `.metadata` (source file info) |

> [!IMPORTANT]
> The `Document` object is fundamental to LangChain. Every loader, splitter, and vector store works with `Document` objects that have:
> - `page_content`: The actual text content
> - `metadata`: A dictionary with source info (filename, page number, URL, etc.)

### 🔄 Alternative Methods

```python
# ---- ALT 1: Plain Python (no LangChain) ----
with open('speech.txt', 'r') as f:
    text = f.read()
# ⚠️ You lose: metadata tracking, chunking integration, batch processing

# ---- ALT 2: PDF Loader ----
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
pages = loader.load()  # Each page becomes a separate Document

# ---- ALT 3: CSV Loader ----
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")
docs = loader.load()  # Each row becomes a Document

# ---- ALT 4: Directory Loader (load ALL files in a folder) ----
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./documents/", glob="**/*.txt")
docs = loader.load()
```

---

## Cell 5 — RAG: Loading a Web Page

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(web_paths=("https://www.databricks.com/blog/what-are-large-language-models",))
print(loader)
print(loader.load())
```

### Why use this?

This loads the **entire HTML page** and converts it to text. Notice in the output that it includes navigation menus, footers, ads — lots of noise!

---

## Cell 6 — RAG: Loading Web Page with BeautifulSoup Filter

```python
import bs4
loader = WebBaseLoader(
    web_paths=("https://www.databricks.com/blog/what-are-large-language-models",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=("post-title","post-content","post-header")
    ))
)
print(loader)
print(loader.load())
```

### What's Different from Cell 5?

| Aspect | Cell 5 (No filter) | Cell 6 (With filter) |
|--------|---------------------|-----------------------|
| Content | Entire page (nav, footer, ads, main content) | Only elements with CSS classes `post-title`, `post-content`, `post-header` |
| Data quality | Noisy — lots of irrelevant text | Clean — only the article content |
| Token cost | Higher (more text = more tokens when sent to LLM) | Lower |

> [!TIP]
> `bs_kwargs` passes keyword arguments directly to BeautifulSoup. `SoupStrainer` tells BS4 to only parse HTML elements matching the specified CSS classes. This is **critical for RAG quality** — garbage in = garbage out!

### 🔄 Alternative Methods

```python
# ---- ALT 1: Using Selenium for JavaScript-rendered pages ----
# pip install selenium
from langchain_community.document_loaders import SeleniumURLLoader
loader = SeleniumURLLoader(urls=["https://example.com"])

# ---- ALT 2: Using the Unstructured library ----
from langchain_community.document_loaders import UnstructuredURLLoader
loader = UnstructuredURLLoader(urls=["https://example.com"])

# ---- ALT 3: Using requests + BeautifulSoup manually ----
import requests
from bs4 import BeautifulSoup
html = requests.get("https://example.com").text
soup = BeautifulSoup(html, 'html.parser')
text = soup.get_text()
```

---

## Cell 7 — Text Splitting: `RecursiveCharacterTextSplitter`

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("speech.txt")
docs = loader.load()

text_splitters = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
final_documents = text_splitters.split_documents(docs)

print(len(final_documents[0].page_content))  # 495
print(len(final_documents[1].page_content))  # 498
print(len(final_documents[2].page_content))  # 439
```

### Why Do We Split Text?

LLMs have a **context window limit** (e.g., GPT-4o-mini ≈ 128K tokens). More importantly, for RAG:
1. We need to find the **most relevant chunk** of text, not the whole document
2. Smaller chunks = more precise search results
3. Embedding models work better on focused text passages

### Parameters Explained

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `chunk_size` | 500 | Maximum 500 characters per chunk |
| `chunk_overlap` | 100 | Each chunk shares 100 characters with the next chunk |

### Why `chunk_overlap`?

```
Without overlap:  [chunk1: "...the Civil War."] [chunk2: "He called for a new..."]
With overlap:     [chunk1: "...the Civil War. He called"] [chunk2: "Civil War. He called for a new..."]
```
Overlap prevents important context from being cut off at chunk boundaries.

### How `RecursiveCharacterTextSplitter` Works

It tries to split using these separators **in order of priority**:
1. `"\n\n"` (double newline — paragraph breaks)
2. `"\n"` (single newline — line breaks)
3. `" "` (space — word boundaries)
4. `""` (empty string — individual characters, last resort)

This ensures chunks break at natural boundaries when possible.

---

## Cell 8 — Text Splitting: `CharacterTextSplitter` (Comparison)

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader("speech.txt")
docs = loader.load()

text_splitters = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
final_documents = text_splitters.split_documents(docs)
```

### Key Difference: `CharacterTextSplitter` vs `RecursiveCharacterTextSplitter`

| Feature | `CharacterTextSplitter` | `RecursiveCharacterTextSplitter` |
|---------|------------------------|----------------------------------|
| Separator | Uses **only ONE** separator you specify | Tries **multiple** separators in priority order |
| Behavior | Splits on `"\n"` only; if a chunk is still > 500 chars, it **keeps it oversized** and shows a warning | Falls back to `" "` or `""` to guarantee chunk_size is respected |
| Warning | `"Created a chunk of size 899, which is longer than the specified 500"` | No warning — it will always respect the limit |
| Best for | When your document has a clear, consistent delimiter | **General purpose — recommended for most use cases** |

> [!WARNING]
> Notice in the output: `CharacterTextSplitter` produced a chunk of 899 characters despite `chunk_size=500`. This is because the text between `\n` separators was longer than 500, and `CharacterTextSplitter` won't split further. **Always prefer `RecursiveCharacterTextSplitter` unless you have a specific reason not to.**

---

## Cell 9 — Side-by-Side Comparison (Same Text)

```python
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text = """Chapter 1: The Beginning
...
"""

# CharacterTextSplitter — only splits on "\n\n"
char_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=0, separator="\n\n")
char_chunks = char_splitter.split_text(text)

# RecursiveCharacterTextSplitter — tries "\n\n", then "\n", then " "
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0)
recursive_chunks = recursive_splitter.split_text(text)
```

### Output Comparison

```
=== CharacterTextSplitter ===
Chunk 3 (153 chars):   ← EXCEEDS the 150 limit! ⚠️
"One day the knight decided to go on a great adventure..."

=== RecursiveCharacterTextSplitter ===
Chunk 3 (147 chars):   ← Respects the 150 limit ✅
"One day the knight decided to go on a great adventure..."
Chunk 4 (5 chars):
"dark."               ← The remaining part goes into a new chunk
```

> [!NOTE]
> Notice `split_text()` is used here instead of `split_documents()`. The difference:
> - `split_text(string)` → takes raw text, returns `list[str]`
> - `split_documents(docs)` → takes `list[Document]`, returns `list[Document]` (preserves metadata)

---

## Cell 10 — HTML Header Text Splitter

```python
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

html_string = """
<!DOCTYPE html>
<html><body>
    <h1>Machine Learning</h1>
    <p>Machine learning is a subset...</p>
    <h2>Supervised Learning</h2>
    <p>Supervised learning uses labeled datasets...</p>
    <h3>Classification</h3>
    ...
</body></html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

# Step 1: Split by HTML headers
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_splits = html_splitter.split_text(html_string)

# Step 2: Further split into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
final_splits = text_splitter.split_documents(html_splits)
```

### Why a Two-Step Split?

1. **Step 1 (`HTMLHeaderTextSplitter`)**: Splits by HTML structure (`<h1>`, `<h2>`) and **preserves the header hierarchy as metadata**
2. **Step 2 (`RecursiveCharacterTextSplitter`)**: Further splits any chunks that are still too large

### The Magic: Metadata Preservation

```
Chunk 4 → content: "Supervised learning uses labeled datasets..."
           metadata: {'Header 1': 'Machine Learning', 'Header 2': 'Supervised Learning'}
```

When you search your vector store later, you know **exactly which section** a chunk came from. This is incredibly valuable for RAG!

### 🔄 Alternative Methods

```python
# ---- ALT 1: Markdown Header Splitter ----
from langchain_text_splitters import MarkdownHeaderTextSplitter
headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)

# ---- ALT 2: Semantic Chunking (AI-powered splitting) ----
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
splitter = SemanticChunker(OpenAIEmbeddings())
# Splits based on meaning changes, not character count
# ⚠️ More expensive (uses embeddings) but much higher quality chunks

# ---- ALT 3: Token-based splitting ----
from langchain_text_splitters import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=20)
# Splits by token count (more accurate for LLM context limits)
```

---

## Cell 11 & 12 — Environment Setup for Embeddings

```python
import os
from dotenv import load_dotenv
load_dotenv()  # load all the environment variables
```

> [!NOTE]
> The commented-out line `os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")` is redundant because `load_dotenv()` already puts the key into `os.environ`. You don't need it.

---

## Cell 13 — OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings 
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
embeddings
```

### What Are Embeddings?

Embeddings convert text into a **list of numbers** (a vector). Texts with similar meanings get similar vectors. This is how "semantic search" works in RAG — instead of keyword matching, we compare vector similarity.

### Parameters Explained

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `model` | `"text-embedding-3-large"` | OpenAI's most powerful embedding model (3072 dimensions natively) |
| `dimensions` | `1024` | Reduce from 3072 → 1024 dimensions to save storage/cost. Quality remains very high |

### Available OpenAI Embedding Models

| Model | Dimensions | Price | Best for |
|-------|-----------|-------|----------|
| `text-embedding-3-small` | 1536 | Cheapest | Budget-friendly projects |
| `text-embedding-3-large` | 3072 (or custom) | Mid-range | Best quality-to-cost ratio |
| `text-embedding-ada-002` | 1536 | Legacy | Deprecated, avoid for new projects |

---

## Cell 14 — Running an Embedding Query

```python
text = "This is a tutorial on OPENAI embedding"
query_result = embeddings.embed_query(text)
query_result
```

### What Happens Here?

1. The text `"This is a tutorial on OPENAI embedding"` is sent to OpenAI's API
2. OpenAI returns a list of 1024 floating-point numbers (because we set `dimensions=1024`)
3. This vector **numerically represents the meaning** of that sentence

### Two Key Methods

| Method | Use Case | Input | Output |
|--------|----------|-------|--------|
| `embed_query(text)` | For a **single** user query | One string | One vector (`list[float]`) |
| `embed_documents([text1, text2, ...])` | For **batch** embedding documents | List of strings | List of vectors |

---

## Cell 15 — Checking Embedding Dimensions

```python
print(len(query_result))  # Output: 1024
```

Confirms the embedding has 1024 dimensions, as we specified.

---

## 🔄 Alternative Embedding Methods

### 1. Hugging Face Embeddings (FREE, runs locally)

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# ✅ Free, runs on your machine
# ✅ No API key needed
# ⚠️ Lower quality than OpenAI for some tasks
# ⚠️ Requires GPU for large-scale embedding
```

### 2. Ollama Embeddings (FREE, runs locally)

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
# ✅ Free, runs locally via Ollama
# ✅ Good quality
# ⚠️ Requires Ollama installed locally
```

### 3. Google Embeddings

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# ✅ High quality
# ⚠️ Requires GOOGLE_API_KEY
```

---

## 🗺️ Complete RAG Pipeline (How Everything Connects)

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAG PIPELINE                              │
│                                                                  │
│  1. LOAD           2. SPLIT            3. EMBED      4. STORE   │
│  ┌──────────┐     ┌──────────────┐    ┌─────────┐  ┌─────────┐ │
│  │TextLoader│ →   │Recursive     │ →  │OpenAI   │→ │ChromaDB │ │
│  │WebLoader │     │CharText      │    │Embedding│  │FAISS    │ │
│  │PDFLoader │     │Splitter      │    │HuggingF │  │Pinecone │ │
│  └──────────┘     └──────────────┘    └─────────┘  └─────────┘ │
│                                                         ↓       │
│  5. QUERY                                                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ User asks question → Embed question → Search vector store │ │
│  │ → Retrieve top-K chunks → Pass to LLM as context         │ │
│  │ → LLM generates answer based on retrieved context         │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

> [!TIP]
> Your notebook covers steps 1-3. The next steps would be:
> - **Step 4**: Store embeddings in a vector database (ChromaDB — you've already installed it!)
> - **Step 5**: Query the vector store and pass results to an LLM

---

## 📚 Summary of All Tools Used

| Category | Tool Used | Alternative |
|----------|-----------|-------------|
| **Agent Framework** | `create_agent` | `create_react_agent` (LangGraph), direct OpenAI function calling |
| **Document Loading** | `TextLoader`, `WebBaseLoader` | `PyPDFLoader`, `CSVLoader`, `DirectoryLoader`, `SeleniumURLLoader` |
| **Text Splitting** | `RecursiveCharacterTextSplitter`, `CharacterTextSplitter`, `HTMLHeaderTextSplitter` | `TokenTextSplitter`, `MarkdownHeaderTextSplitter`, `SemanticChunker` |
| **Embeddings** | `OpenAIEmbeddings` | `HuggingFaceEmbeddings`, `OllamaEmbeddings`, `GoogleGenerativeAIEmbeddings` |
| **LLM** | `ChatOpenAI (gpt-4o-mini)` | `ChatAnthropic`, `ChatGoogleGenerativeAI`, `Ollama` |
