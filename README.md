# Efficient Retrieval-Augmented Generation using Small Language Model

This repository contains the final project for **CSCI 611 - Spring 2025**, implemented by **Group 3**.

## 📌 Objective

This project demonstrates an offline Retrieval-Augmented Generation (RAG) pipeline using lightweight models (MiniLM + DistilGPT2). It enables accurate, low-latency answers to user queries by combining semantic search with local language generation.

---

## 👥 Team Members

- Pavan Sesha Sai Kasukurthi  
- Lokesh Repala  
- Udaychandra Gollapally  
- Sai Rikwith Daggu  

---

## 🧠 Key Features

- Fully **offline and local** RAG pipeline  
- Efficient **dense vector search** using PyTorch  
- **Sliding window chunking** to preserve semantic continuity  
- Built using **MiniLM** for embedding and **DistilGPT2** for generation  
- Tested on a **1,200-page nutrition textbook**  

---

## 🚀 How to Run

### ⚙️ IMPORTANT: Enable GPU in Google Colab
Before running this notebook:
1. Go to **Runtime > Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

---

### ▶️ Run the Code

Open the Jupyter Notebook:
```bash
CSCI611_Group3_Code.ipynb
````

Run the cells from top to bottom. Everything is included — from PDF extraction to final answer generation — with all dependencies and API token handling built-in.

---

## 🔐 Hugging Face API Token Required

To run the generation step using the Hugging Face `transformers` pipeline, you need a Hugging Face API token.

### Options:

1. **Use your own token** — Get one from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Or use our shared token**:

```
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
```

### In Google Colab:

Go to the left-side **🔐 Secrets tab** → click **“+ Add a new secret”**
Add:

* **Name:** `HF_TOKEN`
* **Value:** (Paste your token here)

The code will automatically load the token using:

```python
import os
token = os.getenv("HF_TOKEN")
```

---

## 🧪 Results

| Version      | Retrieval      | Generator  | Accuracy | Hallucination |
| ------------ | -------------- | ---------- | -------- | ------------- |
| Initial      | None           | GPT2       | \~40%    | Very High     |
| Intermediate | Keyword        | GPT2       | \~60%    | Medium        |
| Final (ours) | Dense (MiniLM) | DistilGPT2 | \~87%    | Low           |

---

## 🖼️ Screenshots and What They Show

### `01_pdf_text_extraction.png`

Shows the output of the PDF ingestion code. It confirms the document is being read and parsed correctly — including page number, token stats, and cleaned text.

### `02_chunking_output.png`

Displays a full sample chunk (\~10 sentences). This confirms your sliding window chunking logic is working and producing meaningful, context-rich text blocks.

### `03_embedding_shape.png`

Outputs the tensor shape of your document embeddings (e.g., `torch.Size([N, 384])`). This proves the dense vectors have been successfully computed and stored in memory.

### `04_topk_retrieval.png`

Shows the query, cosine similarity scores, and matched text chunks retrieved. This verifies the semantic search mechanism is accurate and grounded in context.

### `05_final_generation.png`

Displays a real generated answer from DistilGPT2 in response to a query, along with the retrieved context. This confirms end-to-end pipeline execution.

---

## 📘 References

* [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
* [Sentence Transformers](https://www.sbert.net/)
* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [PyMuPDF](https://pymupdf.readthedocs.io/)
* [LangChain](https://www.langchain.com/)

---

## 🔭 Future Work

* Add hybrid BM25 + dense retriever
* Create a lightweight UI (Gradio/Streamlit)
* Explore multi-modal documents
* Benchmark with academic QA datasets
