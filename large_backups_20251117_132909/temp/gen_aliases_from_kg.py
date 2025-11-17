import csv, re, json
from pathlib import Path

KG  = r"F:\graph-corag-clean\data\kg_edges.merged.plus.csv"
OUT = r"F:\graph-corag-clean\config\aliases.manual.jsonl"

def normalize_key(k): return (k or "").strip().lower().lstrip("\ufeff")
def pick(keys, *cands): 
    s=set(keys); 
    return next((c for c in cands if c in s), None)

def looks_entitylike(k):
    return bool(re.search(r'^(drug_|disease_|chemical_|gene_)|^(rxnorm:|chebi:|drugbank:)', k or "", re.I))

def surface_from_id(k):
    s = re.sub(r'^(drug_|disease_|chemical_|gene_)', '', k or "", flags=re.I)
    s = s.replace('_',' ').replace('-', ' ')
    s = re.sub(r'\s+',' ', s).strip()
    return s

seen=set(); rows=[]
with open(KG, encoding="utf-8-sig", newline="") as f:
    rdr = csv.DictReader(f)
    rdr.fieldnames = [normalize_key(x) for x in rdr.fieldnames]
    head_key = pick(rdr.fieldnames, "head","source","h")
    tail_key = pick(rdr.fieldnames, "tail","target","t")
    if not head_key or not tail_key:
        raise RuntimeError(f"Cannot find head/tail in columns: {rdr.fieldnames}")

    for row in rdr:
        row = {normalize_key(k): v for k,v in row.items()}
        for k in (row.get(head_key), row.get(tail_key)):
            if not k or k in seen: continue
            if looks_entitylike(k):
                s = surface_from_id(k)
                if s and not s.isdigit() and len(s)>=3:
                    rows.append({"kg_id": k, "synonyms": [s]})
                    seen.add(k)

Path(OUT).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
print(f"Wrote {len(rows)} aliases → {OUT}")
