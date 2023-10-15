import gradio as gr

import copy
import random
import os
import requests
import time
import sys

from huggingface_hub import snapshot_download
from llama_cpp import Llama

# Initial system prompt
SYSTEM_PROMPT = ("Your name is ANIMA, an Advanced Nature Inspired Multidisciplinary Assistant, and a leading expert "
                "in biomimicry, biology, engineering, industrial design, environmental science, physiology, and paleontology. "
                "Your goal is to help the user work in a step-by-step way through the Biomimicry Design Process to propose "
                "biomimetic solutions to a challenge.")

# Download model snapshot
repo_name = "Severian/ANIMA-Phi-Neptune-Mistral-7B-gguf"
model_name = "anima-phi-neptune-mistral-7b.Q4_K_M.gguf"
snapshot_download(repo_id=repo_name, local_dir=".", allow_patterns=model_name)

# Initialize Llama model
model = Llama(
    model_path=model_name,
    n_ctx=2000,
    n_parts=1,
)

# Maximum number of new tokens to generate
max_new_tokens = 1500

# User message function
def user(message, history):
    new_history = history + [[message, None]]
    return "", new_history

# Bot message function
def bot(history, system_prompt, top_p, top_k, temp):
    tokens = model.tokenize(system_prompt.encode("utf-8"))
    
    for user_message, bot_message in history[:-1]:
        message_tokens = model.tokenize(user_message.encode("utf-8"))
        tokens.extend(message_tokens)
        
        if bot_message:
            message_tokens = model.tokenize(bot_message.encode("utf-8"))
            tokens.extend(message_tokens)

    last_user_message = history[-1][0]
    message_tokens = model.tokenize(last_user_message.encode("utf-8"))
    tokens.extend(message_tokens)

    generator = model.generate(tokens, top_k=top_k, top_p=top_p, temp=temp)

    partial_text = ""
    for i, token in enumerate(generator):
        if token == model.token_eos() or (max_new_tokens is not None and i >= max_new_tokens):
            break
        partial_text += model.detokenize([token]).decode("utf-8", "ignore")
    history[-1][1] = partial_text
    return history

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    favicon = '<img src="https://cdn-uploads.huggingface.co/production/uploads/64740cf7485a7c8e1bd51ac9/seIR5ErFdX5Snr4O7r7tY.png" width="48px" style="display: inline">'
    gr.Markdown(f"""<h1><center>{favicon}A N I M A</center></h1>ANIMA is an expert in various scientific disciplines.""")
    with gr.Row():
        with gr.Column(scale=5):
            system_prompt = gr.Textbox(label="System Prompt", placeholder="", value=SYSTEM_PROMPT, interactive=False)
            chatbot = gr.Chatbot(label="Dialogue").style(height=400)
        with gr.Column(min_width=80, scale=1):
            with gr.Tab(label="Generation Parameters"):
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, interactive=True, label="Top-p")
                top_k = gr.Slider(minimum=10, maximum=100, value=30, step=5, interactive=True, label="Top-k")
                temp = gr.Slider(minimum=0.0, maximum=2.0, value=0.01, step=0.01, interactive=True, label="Temperature")
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(label="Send Message", placeholder="Send Message", show_label=False).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Send")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    with gr.Row():
        gr.Markdown("""WARNING: The model may generate factually or ethically incorrect texts. We are not responsible for this.""")
    
    submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).success(fn=bot, inputs=[chatbot, system_prompt, top_p, top_k, temp], outputs=chatbot, queue=True)
    submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).success(fn=bot, inputs=[chatbot, system_prompt, top_p, top_k, temp], outputs=chatbot, queue=True)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, submit_click_event], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=128, concurrency_count=1)
demo.launch()
