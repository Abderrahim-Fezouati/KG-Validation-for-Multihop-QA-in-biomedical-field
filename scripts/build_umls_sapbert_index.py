import json, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom

def _dedupe_pairs(pairs):
    seen, out = set(), []
    for cui, name in pairs:
        if not isinstance(cui, str): cui = str(cui)
        if not isinstance(name, str): continue
        cui = cui.strip(); name = name.strip()
        if not cui or not name: continue
        key = (cui, name.lower())
        if key not in seen:
            seen.add(key)
            out.append((cui, name))
    return out

def load_dict(path):
    """
    Accepts multiple overlay shapes and returns a list of (cui, name).
    Supported:
      - {"CUI": {"names":[...], "synonyms":[...]} , ...}
      - {"name": "CUI", ...}
      - {"name": ["CUI1","CUI2"], ...}
      - [{"cui":"Cxxx","name":"..."}, ...]
      - [{"CUI":"Cxxx","names":[...]} , ...]
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = []

    def add(cui, name):
        if cui is None or name is None: return
        pairs.append((cui, name))

    if isinstance(raw, dict):
        # Case A: CUI -> obj with names/synonyms
        if all(isinstance(v, dict) for v in raw.values()):
            for cui, obj in raw.items():
                names = []
                for k in ("names","synonyms","terms","aliases"):
                    v = obj.get(k, [])
                    if isinstance(v, str): v = [v]
                    if isinstance(v, list): names.extend(v)
                for n in names:
                    add(cui, n)
        else:
            # Case B: name -> CUI or name -> [CUIs]
            for name, cui_or_list in raw.items():
                if isinstance(cui_or_list, list):
                    for cui in cui_or_list:
                        add(cui, name)
                else:
                    add(cui_or_list, name)

    elif isinstance(raw, list):
        # Case C: list of dict entries
        for item in raw:
            if not isinstance(item, dict): continue
            if "cui" in item and "name" in item:
                add(item["cui"], item["name"])
            elif "CUI" in item and ("names" in item or "synonyms" in item):
                cui = item["CUI"]
                names = item.get("names", []) or item.get("synonyms", [])
                if isinstance(names, str): names = [names]
                for n in names:
                    add(cui, n)
            else:
                # last resort: try to find fields heuristically
                cui = item.get("cui") or item.get("CUI")
                names = item.get("names") or item.get("synonyms") or item.get("name")
                if isinstance(names, str): names = [names]
                if isinstance(names, list):
                    for n in names: add(cui, n)

    # Clean + dedupe
    pairs = _dedupe_pairs(pairs)
    print(f"[load_dict] normalized to {len(pairs)} (cui, name) pairs")
    return pairs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = load_dict(args.dict)
    print(f"Loaded {len(pairs)} usable pairs")

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    mdl = AutoModel.from_pretrained(args.model_name).to(args.device).eval()

    embs = []
    rows = []
    with torch.no_grad():
        for i in range(0, len(pairs), args.batch_size):
            chunk = pairs[i:i+args.batch_size]
            texts = [name for _, name in chunk]
            x = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(args.device)
            outp = mdl(**x)
            pool = mean_pool(outp.last_hidden_state, x["attention_mask"])
            pool = torch.nn.functional.normalize(pool, p=2, dim=1)  # cosine via L2-normalized
            embs.append(pool.cpu().numpy())
            rows.extend(chunk)
            if (i // args.batch_size) % 50 == 0:
                print(f"Encoded {i+len(chunk)}/{len(pairs)}")

    import numpy as np, faiss
    embs = np.vstack(embs).astype("float32")
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss.write_index(index, str(out / "index.faiss"))
    print("Wrote index.faiss")

    row_map = [{"cui": cui, "name": name} for (cui, name) in rows]
    (out / "cui_map.json").write_text(json.dumps(row_map, ensure_ascii=False), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"model": args.model_name, "dim": d}, indent=2), encoding="utf-8")
    print("Done.")
if __name__ == "__main__":
    main()
