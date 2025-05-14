# 🛍️ Defacto Customer Service Chatbot (Facebook Messenger)

This project is a production-ready, AI-powered chatbot designed for **Defacto**, built using **Flask**, **LangChain**, **FAISS**, and **HuggingFace Transformers**. The chatbot integrates seamlessly with **Facebook Messenger**, providing automated customer support by answering frequently asked questions and product-related inquiries using document-based retrieval.

## ✨ Features

- ✅ Facebook Messenger integration using Webhooks  
- 📚 Document-based Q&A powered by LangChain  
- 🧠 Semantic search using FAISS and HuggingFace sentence embeddings  
- 🔁 Contextual message handling with session support  
- 🌐 Arabic and English message processing  
- ⚙️ Easy deployment with Flask  

---

## 🛠️ Tech Stack

| Component         | Description                                 |
|------------------|---------------------------------------------|
| Flask            | Python web framework (for webhook/API)       |
| LangChain        | Framework for LLM-based applications         |
| FAISS            | Fast vector similarity search                |
| HuggingFace      | Sentence transformers for embedding texts    |
| Facebook Graph API | Messenger webhook handling & messaging     |

---

## 📦 Project Structure

defacto-chatbot/
├── app.py # Main Flask app
├── vectorstore_faiss/ # FAISS vector DB
├── data/ # Source documents (FAQs, etc.)
├── logs/ # Session and interaction logs
├── utils/
│ ├── embedding.py # HuggingFace embedding setup
│ ├── retrieval.py # Retrieval logic using FAISS
│ ├── keyword_utils.py # Keyword extraction and filtering
│ └── session_manager.py # Facebook user session handling
├── requirements.txt
└── README.md

