import json, faiss
from pathlib import Path
root = Path(r"F:\graph-corag-clean\artifacts\concept_index")
for et in ["drug","disease","protein"]:
    rows_p = root/et/"rows.jsonl"
    faiss_p = root/et/"index.faiss"
    if not rows_p.exists() or not faiss_p.exists():
        print(f"[{et}] MISSING")
        continue
    rows = [json.loads(l) for l in rows_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    index = faiss.read_index(str(faiss_p))
    print(f"[{et}] rows={len(rows)}  faiss.ntotal={index.ntotal}")
