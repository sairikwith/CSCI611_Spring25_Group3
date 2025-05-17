# Efficient Retrieval-Augmented Generation using Small Language Model

This repository contains the final project for **CSCI 611 - Spring 2025**, implemented by **Group 3**.

## Objective

This project demonstrates an offline Retrieval-Augmented Generation (RAG) pipeline using lightweight models (MiniLM + DistilGPT2). It enables accurate, low-latency answers to user queries by combining semantic search with local language generation.

---

## Team Members

- Pavan Sesha Sai Kasukurthi  
- Lokesh Repala  
- Udaychandra Gollapally  
- Sai Rikwith Daggu  

---

## Key Features

- Fully **offline and local** RAG pipeline  
- Efficient **dense vector search** using PyTorch  
- **Sliding window chunking** to preserve semantic continuity  
- Built using **MiniLM** for embedding and **DistilGPT2** for generation  
- Tested on a **1,200-page nutrition textbook**  

---

##  How to Run

Open the Jupyter Notebook:

```bash
CSCI611_Group3_Code.ipynb
````

Run the cells from top to bottom. Everything is included — from PDF extraction to answer generation — with all dependencies loaded within the notebook itself.

---

### Hugging Face API Token Required

To run the generation step using the Hugging Face `transformers` pipeline, you need a Hugging Face API token.

### Options:

1. **Use your own token** — Get one from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. **Or use our shared token**:

```
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
```

### In Google Colab:

Go to the left-side ** Secrets tab** → click **“+ Add a new secret”**
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
