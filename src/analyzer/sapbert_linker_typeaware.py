import json
from pathlib import Path
import numpy as np
import torch, faiss
from transformers import AutoTokenizer, AutoModel
from .normalizer import NameNormalizer

class TypeAwareLinker:
    def __init__(self, model_name: str, index_root: str, norm_cfg: str, device: str=None, max_len:int=64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.mdl = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.norm = NameNormalizer(norm_cfg)
        self.root = Path(index_root)
        self._load_indices()

    def _load_indices(self):
        self.indices={}
        self.rows={}
        for sub in self.root.iterdir():
            if not sub.is_dir(): continue
            idxp = sub / "index.faiss"
            rowp = sub / "rows.jsonl"
            if idxp.exists() and rowp.exists():
                self.indices[sub.name] = faiss.read_index(str(idxp))
                items=[]
                with open(rowp, "r", encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        if line:
                            items.append(json.loads(line))
                self.rows[sub.name] = items

    @torch.no_grad()
    def _encode(self, mention:str):
        texts = self.norm.normalize_list(mention) or [mention]
        x = self.tok([texts[0]], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        out = self.mdl(**x).last_hidden_state
        mask = x["attention_mask"].unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype("float32")

    def link(self, mention:str, expected_types:list[str], topk:int=8):
        q = self._encode(mention)
        results=[]
        types = expected_types or list(self.indices.keys())
        for t in types:
            if t not in self.indices: continue
            sims, idx = self.indices[t].search(q, topk)
            sims, idx = sims[0], idx[0]
            rows = self.rows[t]
            for s,i in zip(sims, idx):
                if i<0: continue
                r = rows[i]
                results.append({"kg_id": r["kg_id"], "name": r["text"], "entity_type": t, "score": float(s)})
        results.sort(key=lambda x: x["score"], reverse=True)
        seen, out = set(), []
        for r in results:
            if r["kg_id"] in seen: continue
            seen.add(r["kg_id"]); out.append(r)
        return out[:topk]







