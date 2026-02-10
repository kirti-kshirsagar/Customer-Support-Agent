# app.py
import math
import os
from typing import Dict, List, Tuple, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

from faq_data import FAQ_DATA

# Load environment variables from .env (for OPENAI_API_KEY)
load_dotenv()

# Simple text similarity functions for FAQ retrieval

def normalize(text: str) -> List[str]:
    """Lowercase and split on whitespace, remove empty tokens."""
    return [t.strip().lower() for t in text.split() if t.strip()]

def bow_vector(tokens: List[str]) -> Dict[str, int]:
    """Bag-of-words frequency vector."""
    vec: Dict[str, int] = {}
    for tok in tokens:
        vec[tok] = vec.get(tok, 0) + 1
    return vec

def cosine_sim(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(a.keys()) & set(b.keys())
    num = sum(a[t] * b[t] for t in common)
    if num == 0:
        return 0.0
    denom_a = math.sqrt(sum(v * v for v in a.values()))
    denom_b = math.sqrt(sum(v * v for v in b.values()))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return num / (denom_a * denom_b)

# Precompute FAQ vectors once at import time for efficiency
FAQ_ENTRIES = []
for item in FAQ_DATA["questions"]:
    q = item["question"]
    a = item["answer"]
    tokens = normalize(q)
    vec = bow_vector(tokens)
    FAQ_ENTRIES.append(
        {
            "question": q,
            "answer": a,
            "vec": vec,
        }
    )

def retrieve_best_faq(user_query: str, threshold: float = 0.65) -> Tuple[Optional[str], float]:
    """
    Return best FAQ answer and similarity score if above threshold.
    Otherwise, return (None, best_score).
    """
    uq_tokens = normalize(user_query)
    uq_vec = bow_vector(uq_tokens)
    if not uq_vec:
        return None, 0.0

    best_answer = None
    best_score = 0.0

    for entry in FAQ_ENTRIES:
        score = cosine_sim(uq_vec, entry["vec"])
        if score > best_score:
            best_score = score
            best_answer = entry["answer"]

    if best_answer and best_score >= threshold:
        return best_answer, best_score
    return None, best_score

# OpenAI fallback for non-FAQ questions

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def call_llm_fallback(user_query: str) -> str:
    """
    Fallback to OpenAI for non-FAQ questions.
    Uses /v1/chat/completions with a simple system+user prompt.
    """
    if not OPENAI_API_KEY:
        return (
            "I don’t have a specific FAQ answer for that, and no OpenAI API key is configured. "
            "Please set OPENAI_API_KEY in your environment."
        )

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful, generic AI assistant for Thoughtful AI customers. "
                            "If the question is about healthcare automation or Thoughtful AI, answer helpfully from general knowledge. "
                            "Otherwise, answer as a normal assistant."
                        ),
                    },
                    {"role": "user", "content": user_query},
                ],
                "max_tokens": 300,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return (
            "I tried to ask OpenAI’s language model but ran into an issue. "
            "Please check your API key, network, or try again later."
        )

# Streamlit chat UI

st.set_page_config(page_title="Thoughtful AI Support Agent", page_icon="💬")

st.title("Thoughtful AI – Customer Support Agent")
st.write(
    "Ask me questions about Thoughtful AI’s agents (EVA, CAM, PHIL, benefits, etc.). "
    "I’ll answer from a predefined FAQ when possible, and use a generic AI assistant for everything else."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box at bottom
user_input = st.chat_input("Type your question here...")
if user_input:
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1) Try FAQ retrieval
    faq_answer, score = retrieve_best_faq(user_input)

    if faq_answer:
        # We found a good FAQ match
        source_note = f"_Answered from Thoughtful AI FAQ (similarity: {score:.2f})._"
        answer_text = faq_answer
    else:
        # 2) Fallback: generic LLM
        source_note = "_No close FAQ match found. Answered by a generic AI assistant._"
        answer_text = call_llm_fallback(user_input)

    full_answer = f"{answer_text}\n\n{source_note}"

    # Store and display assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    with st.chat_message("assistant"):
        st.markdown(full_answer)
