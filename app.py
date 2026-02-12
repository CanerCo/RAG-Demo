import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


import numpy as np
import json

vectors = np.load("./data/embeddings.npy")

with open('./data/chunks.json', "r") as f:
    chunks_list = json.load(f)

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def retrieve(query, vectors, chunks_list, model):
    q_vec = model.encode([query])[0]
    # Compute cosine similarty between q_vec and all chunk vectors
    scores = np.dot(vectors, q_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec) + 1e-9)
    top_indx = int(np.argmax(scores))
    return chunks_list[top_indx], scores[top_indx]

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=500)
    return tokenizer.decode(outputs[0], skip_special_tokens = True)


def answer_query(query):
    context = retrieve(query, vectors, chunks_list, embedding_model)
    prompt =  f"""
                You are a QA assistant.

                Rules:
                - Use the context as the ONLY source of factual information.
                - You may paraphrase and combine details into your own sentences.
                - Do NOT add new facts that are not supported by the context.
                - If the context does not contain the answer, say exactly: I do not know.

                Task:
                Answer the question in your own words.

                Context:
                {context}

                Question: {query}

                Answer:"""
    answer = generate_answer(prompt)
    return answer


st.title("RAG QA System")
st.write("Ask a question and get an answer from the documents.")

query = st.text_input("Your question:")
if query:
    answer = answer_query(query)
    st.write("**Answer:**" + answer)

