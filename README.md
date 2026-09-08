# 🦜🔗 LangChain Training

A hands-on learning repository for **LangChain**, **LangGraph**, and the broader LLM ecosystem — covering everything from basic prompt engineering to advanced RAG pipelines with vector stores.

---

## 📁 Project Structure

```
Lang_Chain_Training/
├── Lang_chain/                        # Core LangChain notebooks & notes
│   ├── first.ipynb                    # LangChain fundamentals
│   ├── first_notebook_detailed_notes.md
│   ├── second.ipynb                   # Intermediate LangChain concepts
│   ├── second_notebook_detailed_notes.md
│   ├── Third.ipynb                    # Advanced LangChain topics
│   ├── third_notebook_detailed_notes.md
│   ├── speech.txt                     # Sample text for document loading
│   └── chroma_db/                     # ChromaDB vector store data
│
├── Lang_chain_2/                      # Extended LangChain experiments
│   └── GettingStarted.ipynb           # Getting started guide
│
├── requirements.txt                   # Project dependencies
├── .env                               # API keys (not tracked by git)
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

---

## 🛠️ Technologies & Libraries Used

| Category | Libraries |
|---|---|
| **LLM Framework** | LangChain, LangGraph, LangSmith |
| **LLM Providers** | OpenAI, Ollama, Google Gemini (via deepagents) |
| **Embeddings** | HuggingFace, Sentence-Transformers |
| **Vector Stores** | ChromaDB, FAISS |
| **Document Loaders** | PyPDF, PyMuPDF, ArXiv |
| **Deep Learning** | PyTorch, TensorFlow, Keras |
| **Data Science** | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| **NLP** | NLTK, Transformers (HuggingFace) |
| **Web** | Flask, Requests, BeautifulSoup4 |

---

## 🚀 Getting Started

### Prerequisites

- **macOS** (Apple Silicon / Intel)
- **Python 3.12+** (required for `deepagents`)
- **Homebrew** (macOS package manager)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/Sandeepkumaramgothu/Lang_Chain_Training.git
cd Lang_Chain_Training
```

### 2. Install Python 3.12 (if not already installed)

```bash
# Check your current Python version
python3 --version

# Install Python 3.12 via Homebrew (if needed)
brew install python@3.12

# Verify installation
/opt/homebrew/bin/python3.12 --version
```

### 3. Create a Virtual Environment

```bash
# Create venv with Python 3.12
/opt/homebrew/bin/python3.12 -m venv myenv

# Activate the environment
source myenv/bin/activate        # zsh / bash
# OR
myenv/bin/activate.ps1           # PowerShell (VS Code default on macOS)
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Set Up API Keys

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
# Add other API keys as needed
```

### 6. Launch Jupyter

```bash
jupyter notebook
# OR open notebooks directly in VS Code
```

---

## 📓 Notebook Overview

### Lang_chain/ (Core Learning Path)

| Notebook | Topics Covered |
|---|---|
| **first.ipynb** | LangChain basics, LLM setup, prompt templates, chains, output parsers |
| **second.ipynb** | Document loaders, text splitters, embeddings, vector stores, RAG |
| **Third.ipynb** | Advanced chains, agents, tools, memory, and conversational AI |

> 💡 Each notebook has a companion `*_detailed_notes.md` file with in-depth explanations.

### Lang_chain_2/

| Notebook | Topics Covered |
|---|---|
| **GettingStarted.ipynb** | Extended experiments and getting started with newer features |

---

## 🔄 Upgrading Python in a Virtual Environment

If you need to upgrade your Python version (e.g., a package requires a newer version), follow this guide. **You cannot upgrade a venv in-place** — you must recreate it.

### Why Can't You Just Upgrade?

A Python virtual environment is **tied to the specific Python binary** that created it. The `python` inside your venv is a symlink pointing to a specific Python version (e.g., `python3.10`). There is no `venv upgrade` command — you must **delete and recreate**.

### Step-by-Step Process

#### Step 1: Back Up Your Current Packages

```bash
# Activate the old environment first
source myenv/bin/activate        # zsh / bash
myenv/bin/activate.ps1           # PowerShell

# Export all installed packages with exact versions
pip freeze > requirements_backup.txt

# Deactivate
deactivate
```

> ⚠️ **Do NOT skip this step!** If you delete the venv without backing up, you'll lose your entire package list.

#### Step 2: Check Available Python Versions

```bash
# See what's installed via Homebrew
brew list --versions | grep python

# Search for available versions
brew search python
```

#### Step 3: Install the Newer Python

```bash
# Example: Install Python 3.12
brew install python@3.12

# Verify
/opt/homebrew/bin/python3.12 --version
```

#### Step 4: Delete the Old Virtual Environment

```bash
rm -rf myenv
```

#### Step 5: Create a New Virtual Environment

```bash
# Use the FULL PATH to the new Python binary
/opt/homebrew/bin/python3.12 -m venv myenv
```

#### Step 6: Activate & Reinstall Packages

```bash
# Activate
source myenv/bin/activate        # zsh / bash
myenv/bin/activate.ps1           # PowerShell

# Upgrade pip
pip install --upgrade pip

# Reinstall all previous packages
pip install -r requirements_backup.txt

# Install any new packages you need
pip install deepagents
```

#### Step 7: Verify

```bash
python --version
# Output: Python 3.12.14 ✅
```

### Quick Reference (All Commands)

```bash
# 1. Back up
source myenv/bin/activate && pip freeze > requirements_backup.txt && deactivate

# 2. Install new Python
brew install python@3.12

# 3. Recreate venv
rm -rf myenv
/opt/homebrew/bin/python3.12 -m venv myenv

# 4. Reinstall everything
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements_backup.txt
```

### Terminal Activation Commands Reference

| Terminal / Shell | Activation Command |
|---|---|
| **zsh / bash** (macOS default terminal) | `source myenv/bin/activate` |
| **PowerShell** (VS Code default on macOS) | `myenv/bin/activate.ps1` |
| **fish** | `source myenv/bin/activate.fish` |
| **cmd.exe** (Windows) | `myenv\Scripts\activate.bat` |
| **PowerShell** (Windows) | `myenv\Scripts\Activate.ps1` |

---

## 📋 Key Concepts

| Concept | Description |
|---|---|
| **Virtual Environment** | An isolated Python installation with its own packages, separate from the system Python |
| **`pip freeze`** | Lists all installed packages with exact versions — used for backup and reproducibility |
| **`requirements.txt`** | A file listing packages to install — `pip install -r` reads this file |
| **Homebrew** | macOS package manager — installs tools like Python to `/opt/homebrew/` |
| **LangChain** | Framework for building LLM-powered applications with chains, agents, and tools |
| **RAG** | Retrieval-Augmented Generation — combining LLMs with external knowledge via vector search |
| **Vector Store** | Database optimized for similarity search on embeddings (ChromaDB, FAISS) |
| **Embeddings** | Numerical representations of text that capture semantic meaning |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-topic`)
3. Commit your changes (`git commit -m 'Add new LangChain topic'`)
4. Push to the branch (`git push origin feature/new-topic`)
5. Open a Pull Request

---

## 📝 License

This project is for educational and personal learning purposes.

---

## 👤 Author

**Sandeep Kumar Amgothu**

- GitHub: [@Sandeepkumaramgothu](https://github.com/Sandeepkumaramgothu)