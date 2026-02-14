import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import gc
from huggingface_hub import hf_hub_download

# --- MODEL ARCHITECTURE ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.w

class LoRA(nn.Module):
    def __init__(self, in_f, out_f, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, rank))
    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B)

class TCNLayer(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size, dilation, lora_rank):
        super().__init__()
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.norm = RMSNorm(d_model)
        
        # In Space, weights are loaded via state_dict, but logic remains Fractal
        self.w_in = nn.Parameter(torch.zeros(2*d_ff, d_model))
        self.w_dw = nn.Parameter(torch.zeros(d_ff, 1, kernel_size))
        self.w_out = nn.Parameter(torch.zeros(d_model, d_ff))
        
        self.lora_in = LoRA(d_model, 2*d_ff, lora_rank)
        self.lora_out = LoRA(d_ff, d_model, lora_rank)
        self.scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        res = x
        x = self.norm(x)
        ag = F.linear(x, self.w_in) + self.lora_in(x)
        a, g = ag.chunk(2, dim=-1)
        a = a.transpose(1, 2)
        a = F.pad(a, (self.padding, 0))
        a = F.conv1d(a, self.w_dw, groups=a.shape[1], dilation=self.dilation)
        a = a.transpose(1, 2)
        y = F.silu(a) * torch.sigmoid(g)
        out = F.linear(y, self.w_out) + self.lora_out(y)
        return res + out * self.scale

class ZetaGrid25B(nn.Module):
    def __init__(self, n_layers=32, d_model=4096, d_ff=16384, ks=3, lora_r=128):
        super().__init__()
        self.emb = nn.Embedding(256, d_model)
        self.pos_emb = nn.Embedding(2048, d_model)
        self.layers = nn.ModuleList([
            TCNLayer(d_model, d_ff, ks, 2**(i % 8), lora_r) for i in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.emb(idx) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return F.linear(x, self.emb.weight)

# --- INFERENCE ENGINE ---

model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model
    if model is not None: return
    
    print("🚀 Loading RTH-LM weights from Hugging Face...")
    try:
        # Placeholder for real hub download
        # repo_id = "RthItalia/Rth-lm-25b"
        # ckpt_path = hf_hub_download(repo_id=repo_id, filename="soul_v1.pt")
        # genome_path = hf_hub_download(repo_id=repo_id, filename="genome_v1.npy")
        
        # For now, we initialize a "Small" 1B version if running on standard Space CPU
        model = ZetaGrid25B(n_layers=8, d_model=1024, d_ff=4096).to(DEVICE)
        model.eval()
        print("✅ Model initialized (Lightweight Demo Mode).")
    except Exception as e:
        print(f"❌ Load error: {e}")

@torch.no_grad()
def generate_rth(prompt, temp, top_k, max_len):
    load_model()
    prompt_bytes = list(prompt.encode('utf-8'))
    idx = torch.tensor([prompt_bytes], dtype=torch.long, device=DEVICE)
    
    output_bytes = []
    for _ in range(max_len):
        logits = model(idx[:, -1024:])
        logits = logits[:, -1, :] / temp
        
        # Top-K
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_byte = torch.multinomial(probs, 1)
        
        idx = torch.cat([idx, next_byte], dim=1)
        output_bytes.append(next_byte.item())
        
        if next_byte.item() == 0: break # EOS
        
    return bytes(output_bytes).decode('utf-8', errors='replace')

# --- GRADIO UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🌌 RTH-LM: Gated TCN Interface")
    gr.Markdown("Direct byte-level generation using the Fractal architecture.")
    gr.Markdown(f"**Hardware:** {DEVICE.upper()} | **Mode:** 1B Demo (Lightweight)")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Prompt", placeholder="Write something...", lines=5)
            with gr.Row():
                temp_slider = gr.Slider(0.1, 1.5, 0.7, label="Temperature")
                k_slider = gr.Slider(1, 100, 40, label="Top-K")
                len_slider = gr.Slider(10, 1000, 150, label="Max Bytes")
            btn = gr.Button("Generate Energy", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="RTH-LM Response", lines=12)
    
    btn.click(generate_rth, inputs=[input_text, temp_slider, k_slider, len_slider], outputs=output_text)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome())
