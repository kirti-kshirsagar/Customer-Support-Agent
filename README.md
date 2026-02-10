# Thoughtful AI – Customer Support Agent (FAQ + LLM Fallback)

This project is a simple customer support AI Agent for **Thoughtful AI**. It answers common questions about Thoughtful AI’s agents (EVA, CAM, PHIL, etc.) from a **hardcoded FAQ**, and falls back to a **generic LLM** (OpenAI) for everything else.

It is implemented in Python with a Streamlit chat UI and is designed to demonstrate practical applied‑AI skills.

---

## Features

- Chat‑style interface using Streamlit’s `st.chat_input` and message history.
- Hardcoded FAQ about Thoughtful AI’s agents in `faq_data.py`.
- Simple cosine similarity search over FAQ questions to find the most relevant answer.
- If no FAQ entry is similar enough, the agent calls **OpenAI Chat Completions** as a generic LLM fallback.
- Clear indication of whether a response came from the FAQ or from the LLM.

---

## Tech Stack

- **Language:** Python 3.10+
- **UI:** Streamlit
- **Retrieval:** Bag‑of‑words + cosine similarity (no heavy frameworks)
- **LLM Fallback:** OpenAI Chat Completions API
- **Config:** Environment variables via `python-dotenv`

---

## Project Structure

```text
Customer-Support-Agent/
├─ app.py          # Main Streamlit app 
├─ faq_data.py     # Hardcoded FAQ dataset
├─ requirements.txt
└─ README.md
```
## Setup and Run (Local)

1. Clone the repo
2. Install the requirements:

`pip install -r requirements.txt`

3. Set the env variable in .env file, add your llm key:

`OPENAI_API_KEY=sk-your-real-key-here`

4. To run the streamlit app:

`streamlit run app.py`

This opens the app in your browser

## How It Works
### 1. FAQ Retrieval

- `faq_data.py` contains the predefined dataset of Thoughtful AI Q&A pairs.
- On startup, `app.py`:
    - Normalizes each FAQ question (lowercase, split on spaces).
    - Builds a simple bag‑of‑words frequency vector.

- For each user message:

    - The user’s question is normalized and converted to a vector.
    - Cosine similarity is computed between the user vector and each FAQ vector.
    - If the best similarity score ≥ a threshold (default `0.65`), the corresponding FAQ answer string is returned.

### 2. LLM Fallback
If no FAQ entry passes the threshold:
- The app calls `call_llm_fallback`, which sends a request to OpenAI’s `/v1/chat/completions` endpoint with:
    - A simple system message (“You are a helpful, generic AI assistant for Thoughtful AI customers…”).
    - The user’s question as a user message.
- The returned LLM answer is displayed along with a note:

    - _No close FAQ match found. Answered by a generic AI assistant._

### 3. Chat UI
- Uses Streamlit’s chat components:
    - `st.chat_input` to capture user messages.
    - `st.session_state.messages` to persist conversation state.
    - `st.chat_message("user" | "assistant")` blocks to render bubbles.

- Each assistant message includes:
    - The actual answer text.
    - A small source note indicating FAQ vs. LLM.

<img width="706" height="489" alt="Screenshot 2026-02-10 at 2 21 08 PM" src="https://github.com/user-attachments/assets/0f6234aa-42ac-4941-9b06-55964081f9ae" />

<img width="720" height="482" alt="Screenshot 2026-02-10 at 2 21 32 PM" src="https://github.com/user-attachments/assets/b35576e2-8969-4112-a774-a776ab5bacd7" />


## Design Choices
- **No LangChain / Chroma / FAISS**: For a 5‑item FAQ, a manual similarity function is clearer and easier to reason about than adding multiple heavy dependencies. It keeps the focus on applied AI integration rather than framework plumbing.
- **Explicit Fallback Logic**: The code clearly separates:
    - FAQ retrieval (retrieve_best_faq)
    - LLM fallback (call_llm_fallback)

    This matches the assignment’s requirement of “use predefined dataset, fallback to generic LLM responses for everything else.”
​
- **LLM-Optional**: The app still behaves sensibly without an API key, which makes it easy to run or demo in environments where secrets are not available.

## Extending the Project
Possible extensions:

- Swap bag‑of‑words for embeddings (e.g.: sentence-transformers) and cosine similarity for more robust matching.

- Log FAQ vs. LLM usage for analytics.

- Add simple “human‑in‑the‑loop” simulation (e.g. flag low‑confidence answers).

- Containerize with Docker and deploy to a small cloud instance.
