import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom

class SapBERTLinker:
    def __init__(self, model_dir_or_name: str, index_dir: str, device: str = None, max_len: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_dir_or_name, use_fast=True)
        self.mdl = AutoModel.from_pretrained(model_dir_or_name).to(self.device).eval()

        import faiss  # faiss-cpu
        self.index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
        self.rows  = json.loads(Path(index_dir, "cui_map.json").read_text(encoding="utf-8"))

    @torch.no_grad()
    def _encode(self, texts):
        x = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        out = self.mdl(**x)
        pool = mean_pool(out.last_hidden_state, x["attention_mask"])
        pool = torch.nn.functional.normalize(pool, p=2, dim=1)  # cosine via L2-normalized vectors
        return pool.cpu().numpy().astype("float32")

    def link(self, mention: str, topk: int = 5):
        q = self._encode([mention])
        sims, idx = self.index.search(q, topk)  # inner product on normalized vectors = cosine similarity
        sims, idx = sims[0], idx[0]
        out = []
        for s, i in zip(sims, idx):
            if i < 0: continue
            r = self.rows[i]
            out.append({"cui": r["cui"], "name": r["name"], "score": float(s)})
        return out