import gradio as gr
from rag_pipeline import chatbot_fn, normalize_gradio_history, answer_query_with_history, LAST_DEBUG


with gr.Blocks() as demo:
    gr.Markdown("## Day 5 RAG + Chat History (with Debug)")
    chat = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask something…")
    clear = gr.Button("Clear")

    with gr.Accordion("Debug (what the model sees)", open=False):
        dbg_retrieval = gr.Textbox(label="Retrieval Query", lines=2)
        dbg_prompt = gr.Textbox(label="Final Prompt", lines=18)

    def respond(message, history):
        # history is a list of {"role": ..., "content": ...} dicts in messages mode
        history_pairs = normalize_gradio_history(history)  # your helper: -> list[(user, assistant)]
        answer = answer_query_with_history(message, history_pairs)

        history = history or []
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]

        return history, "", LAST_DEBUG.get("retrieval_query",""), LAST_DEBUG.get("prompt","")

    msg.submit(respond, [msg, chat], [chat, msg, dbg_retrieval, dbg_prompt])
    clear.click(lambda: ([], "", "", ""), None, [chat, msg, dbg_retrieval, dbg_prompt])

demo.launch()




# --- Gradio ----


