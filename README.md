# 📚 Document Intelligence Bot

*Chat with your PDFs, Word docs, and text files—powered by Llama 3 on Groq, LangChain, and Streamlit.*

---

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-f06) ![License](https://img.shields.io/badge/License-MIT-green) ![Made with ❤](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red)

## ✨ Overview

Document Intelligence Bot lets you upload multiple documents (**PDF**, **DOCX**, **TXT**) and query them conversationally. Under the hood it

* splits documents into semantic chunks, embeds them with `sentence‑transformers`,
* stores them in a **FAISS** vector index,
* retrieves relevant chunks at question time via **LangChain’s** Conversational Retrieval Chain, and
* streams answers from blazing‑fast **Llama 3** (or DeepSeek) models served by **Groq API**, complete with inline source citations.

## 🚀 Live Demo

> Feel free to clone and run locally. A public demo link will be added soon.

![demo‑gif](docs/demo.gif)

## 🔑 Key Features

* **Multi‑document support** – batch‑upload PDFs, TXT, or DOCX files
* **Natural‑language Q\&A** – chat like you would with a human tutor
* **Source transparency** – expand any answer to see the exact excerpts used
* **Model choice** – toggle between `llama-3.3‑70b‑versatile`, `llama-3.1‑8b‑instant`, and `deepseek‑r1‑distill‑llama‑70b`
* **Fully local front‑end** – quick, light Streamlit UI

## 🛠️ Tech Stack

| Layer        | Tools                                       |
| ------------ | ------------------------------------------- |
| UI           | Streamlit 1.34†                             |
| LLM          | Groq API (Llama 3 & DeepSeek)               |
| Retrieval    | LangChain v0.3 ConversationalRetrievalChain |
| Embeddings   | `sentence‑transformers/all‑MiniLM‑L6‑v2`    |
| Vector Store | FAISS                                       |

† Any recent Streamlit version ≥1.28 should work.

## 📂 Project Structure

```
├── app.py              # main Streamlit app
├── requirements.txt    # Python deps
├── README.md           # you’re here ✨
└── docs/
    └── demo.gif        # optional demo screencast
```

## ⚡ Quick Start

```bash
# 1️⃣ Clone
$ git clone https://github.com/<your‑user>/document-intelligence-bot.git
$ cd document-intelligence-bot

# 2️⃣ Install deps (Python ≥ 3.9)
$ pip install -r requirements.txt

# 3️⃣ Set your Groq API key
$ export GROQ_API_KEY="sk‑..."  # Linux / macOS
$ setx GROQ_API_KEY "sk‑..."     # Windows

# 4️⃣ Run
$ streamlit run app.py
```

Then open the provided local URL, upload some docs, and start chatting!

## 📝 Environment Variables

| Variable       | Purpose                                                                                 |
| -------------- | --------------------------------------------------------------------------------------- |
| `GROQ_API_KEY` | **Required.** Your secret key from [https://console.groq.com](https://console.groq.com) |

You can also enter the key in the sidebar at runtime, but exporting it avoids re‑typing.

## 💡 Use Cases

* Quickly summarise lengthy research papers or policy docs
* Extract obligations, dates, and parties from contracts
* Create study aids from lecture notes
* Generate FAQ answers from technical manuals


## 🖼️ Screenshots

![Capture](https://github.com/user-attachments/assets/53e79df7-a3ee-4fee-bd66-fc6258d6e5fa)
![Capture2](https://github.com/user-attachments/assets/4e251582-9ba6-406b-a422-3e3c033fafc3)
![Capture3](https://github.com/user-attachments/assets/61953d0f-e610-456d-a83a-41d72943fa58)
---


## 🤝 Contributing

Pull requests are welcome! Open an issue first to discuss major changes.

1. Fork → Create branch → Commit → Push → PR.
2. Follow PEP‑8 and conventional commit messages.
3. Run `pre‑commit` hooks before pushing.

## 📄 License

This project is proprietary and confidential. All rights reserved.

```
© 2025 HUSSAIN ALI. This code may not be copied, modified, distributed, or used without explicit permission.
```

---

## 📬 Contact

For questions or collaboration requests:

* 📧 Email: [choudaryhussainali@outlook.com](mailto:choudaryhussainali@outlook.com)
* 🌐 GitHub: [choudaryhussainali](https://github.com/choudaryhussainali)


## 🙏 Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain) for the retrieval framework
* [Groq](https://console.groq.com) for ultra‑fast inference
* [Sentence‑Transformers](https://www.sbert.net) for lightweight embeddings

---

> ✨ *“Information is only useful when it can be questioned.”* – Adapted from McNamara

Happy chatting! 
