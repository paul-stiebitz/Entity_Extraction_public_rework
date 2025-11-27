# 🧠 Entity Extraction Playground

**Entity Extraction Playground** is an **AI-powered tool for extracting entities from emails**. It provides **streaming extraction**, **batch processing**, and an **interactive web interface** using **Streamlit**, powered by the **Granite 3.3 8B model** via **Ollama**.  

The project demonstrates **parallel processing**, **token-by-token streaming**, and **performance measurement** for different operation modes.

---

## 📦 Main Components

- **Streamlit** – Interactive web interface for extraction and visualization  
- **OpenAI API / Ollama** – Local model `granite3.3:8b` for entity extraction  
- **Python 3.11 + Micromamba** – Lightweight environment for reproducible setup  
- **Parallelization** – Batch extraction of multiple emails via `ThreadPoolExecutor`  
- **Performance Measurement** – Comparison of streaming vs. batch processing  

---

## 🐍 Installation

### 1. Install Micromamba
```bash
cd ~
curl -Ls https://micro.mamba.pm/api/micromamba/linux-ppc64le/latest | tar -xvj bin/micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba --version
```

### 2. Create Python environment
```bash
micromamba create -n entity_extraction_env python=3.11
micromamba activate entity_extraction_env
```

### 3. Install dependencies
```bash
pip install streamlit openai
```

---

## ⚙️ Run the Application
```bash
streamlit run entity_extraction.py --server.port 7860
```

Open the web interface in your browser:
```
http://<your_machine_IP>:7860
```

---

## 📝 Features

### 1. Input Methods
- **Manual Paste:** Enter email text directly into the text field  
- **Load File:** Single `.txt` file with one email  
- **Batch Processing:** Multiple emails from a file (`---` as separator)  

### 2. Entity Types
Default supported types:  
`Person`, `Orders`, `Organization`, `Date`, `Time`, `Location`, `Money`, `Product`  

- Users can select any combination of types  
- Extraction is output in **valid JSON format**  

### 3. Streaming Extraction
- Token-by-token output for **live streaming**  
- Syntax highlighting for JSON (`Keys`, `Strings`, `Numbers`, `Booleans`)  

### 4. Batch Processing
- Parallel extraction using **ThreadPoolExecutor**  
- Progress bar in Streamlit  
- Results sorted by email index  

### 5. Performance Measurement
- Compare **streaming** vs. **batch** processing  
- Supports SMT values: 2, 4, 8  
- Results saved to `performance.txt`  

---

## 🔧 Configuration

- **OpenAI Client:** configured locally via Ollama
```python
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)
```
- **Max Retries / Timeout:** adjustable via `MAX_RETRIES` and `OLLAMA_TIMEOUT`  

---

## 📊 Notes

- Supports **local models** via Ollama (e.g., Granite 3.3 8B)  
- Real-time streaming provides **instant feedback**  
- Batch processing is **scalable** and reduces runtime for many emails  

---

## 🔗 Resources
- [Ollama Documentation](https://ollama.com/docs)  
- [Streamlit Documentation](https://docs.streamlit.io)  
- [Granite Model Info (Hugging Face)](https://huggingface.co/models)

