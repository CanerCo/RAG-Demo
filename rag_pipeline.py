import re 
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json

NEW_SYSTEM_PROMPT =(
    "Answer the user's query using ONLY the CONTEXT and CHAT HISTORY below"
    "Use CHAT HISTORY to resolve references like (e.g., it , they , them etc.)"
    "If the answer is not in the context or CHAT HISTORY, say 'I do not know'."
)

LAST_DEBUG = {"prompt": "", "retrieval_query": ""}



# Load the vectors
vectors = np.load("./data/embeddings.npy")

with open('./data/chunks.json', "r") as f:
    chunks_list = json.load(f)


# Chunking implementation - Day 2 Task 1
def chunk_text(text, max_length=500):
    # Text is splitted into chunks at most max_length characters, at sentence boundaries if possible
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip()) # split on sentence end
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1  <= max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# Test the retrieval on a Simple Query - Day 2 Task 4
def retrieve(query, vectors, chunks_list, model):
    '''Retrieve the most relevant chunk based on cosinle similarity'''
    q_vec = model.encode([query])[0]
    # Compute cosine similarty between q_vec and all chunk vectors
    scores = np.dot(vectors, q_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec) + 1e-9)
    top_indx = int(np.argmax(scores))
    return chunks_list[top_indx], scores[top_indx]

def build_prompt(context, history, question, max_turns):
    # keep last N turns to avoid prompt bloat
    recent = history[-max_turns:] if history else []

    history_block = ""
    for q,a in recent:
        history_block += f"User {q}\n Asisstant: {a}\n"

    return (
        f"{NEW_SYSTEM_PROMPT}\n\n"
        f"[CONTEXT]\n{context}\n\n"
        f"[CHAT HISTORY]\n{history_block if history_block else '(none)'}\n\n"
        f"[CURRENT QUESTION]\nUser: {question}\n Assistant:"
    )

#### MODELS #####
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_name = "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Same with generator pipeline
def generate_answer(prompt):
    """Generate  Answer using FLAN-T5"""
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens = True)


def make_retrieval_query(question: str, history: list[tuple[str, str]]) -> str:
    if not history:
        return question
    last_q, _ = history[-1]
    return f"{last_q}\nFollow-up: {question}"

def answer_query_with_history(question, history):
    retrieval_query = make_retrieval_query(question, history) 
    # retrieve the context
    context = retrieve(query=question, vectors=vectors,chunks_list=chunks_list,model=embedding_model)
    prompt = build_prompt(context=context, history=history, question=question, max_turns=3)

    # store for UI debugging
    LAST_DEBUG["prompt"] = prompt
    LAST_DEBUG["retrieval_query"] = retrieval_query

    out = generate_answer(prompt)
    return out


def _content_to_text(content):
    '''
    Gradio 6+ uses OpenAI-style structured content blocks.
    Older versions often use plain strings.
    '''
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # list of blocks, e.g., [{"type":"text", "text":"hi"}]
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return content ["text"]
    return str(content)


def normalize_gradio_history(history):
    """
    Supports:
    - v4/v5 style: [[user,bot], ...]
    - v6 messages style: [{"role":"user", "content":[...]}, {"role":"assistant","content":[...]} , ...]
    Returns: list[tuple[user_text, assistant_text]]
    """
    if not history:
        return []
    
    # Pair format
    if isinstance(history, list) and history and isinstance(history[0], (list, tuple)) and len(history[0]) == 2:
        out = []
        for u, a in history:
            out.append((_content_to_text(u), _content_to_text(a)))
        return out
    
    # Messages format
    if isinstance(history, list) and history and isinstance(history[0], dict) and "role" in history[0]:
        pairs = []
        pending_user = None
        for msg in history:
            role = msg.get("role")
            text = _content_to_text(msg.get("content"))
            if role == "user":
                pending_user = text
            elif role == "assistant" and pending_user is not None:
                pairs.append((pending_user, text))
                pending_user = None
        return pairs
    
    # Fallback
    return []

def chatbot_fn(message, history):
    history_pairs = normalize_gradio_history(history)

    answer = answer_query_with_history(message, history_pairs)

    return answer


