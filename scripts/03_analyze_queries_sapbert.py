import io, os, re, json, argparse
from collections import defaultdict

def load_umls_dict(path):
    # Expect "surface<tab>cui" per line (surface examples: "adalimumab", "hidradenitis suppurativa")
    entries = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            parts = re.split(r'\t+', line)
            if len(parts) < 2: continue
            surface, cui = parts[0].strip().lower(), parts[1].strip()
            if surface:
                entries.append((surface, cui))
    return entries

REL_PATTERNS = [
    ("CONTRAINDICATED_FOR", r"\bcontraindicat(?:ed|ion|ed for|ed in|ed with)\b|\bnot (?:for|with)\b"),
    ("ADVERSE_EFFECT",      r"\b(adverse|side effect|toxicity|harm|harms|a[e]s?)\b|\bwhat .* effects?\b|\blist .* effects?\b"),
    ("INTERACTS_WITH",      r"\b(interact|interaction|ddi|co-?admin(?:istration)?)\b"),
]

def detect_relation(text: str) -> str:
    t = text.lower()
    for rel, pat in REL_PATTERNS:
        if re.search(pat, t): return rel
    if re.search(r"\b(do|does|is|are|can)\b.*\b(with|together|interact)\b", t.lower()):
        return "INTERACTS_WITH"
    return ""

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("_", " ").strip().lower())

def spans_in_text(cuis, surface_map, text):
    # returns list of (cui, start_index) by earliest surface occurrence
    hits = []
    for surf, cui in surface_map:
        i = text.find(surf)
        if i >= 0:
            hits.append((cui, i))
    hits.sort(key=lambda x: x[1])
    # deduplicate by cui preserving order
    seen = set(); out=[]
    for cui, pos in hits:
        if cui not in seen:
            seen.add(cui); out.append(cui)
    return out

def choose_head_tail(question, rel, umls_entries):
    t = normalize(question)
    # partition dict into drug and disease by CUI prefix
    drug_map    = [(surf, cui) for (surf, cui) in umls_entries if cui.startswith("drug_")]
    disease_map = [(surf, cui) for (surf, cui) in umls_entries if cui.startswith("disease_")]

    drugs   = spans_in_text([], drug_map, t)
    dis     = spans_in_text([], disease_map, t)

    if rel == "INTERACTS_WITH":
        h = drugs[0] if len(drugs) > 0 else ""
        t2 = drugs[1] if len(drugs) > 1 else ""
        return h, t2

    # ADVERSE_EFFECT / CONTRAINDICATED_FOR → (drug, disease)
    h = drugs[0] if drugs else ""
    d = dis[0] if dis else ""
    return h, d

def detect_intent(text):
    t = text.strip().lower()
    if re.match(r"^\s*(do|does|is|are|can)\b", t): return "yesno"
    if re.search(r"\b(what|which|list).*(effect|effects|adverse|side)", t): return "list"
    return "lookup"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", required=True)
    ap.add_argument("--input", required=True)  # raw jsonl: {"qid","question"}
    ap.add_argument("--out", required=True)    # structured jsonl
    args = ap.parse_args()

    umls_entries = load_umls_dict(args.dict)
    n_in, n_out = 0,0

    with io.open(args.input, "r", encoding="utf-8") as fin, \
         io.open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line=line.strip()
            if not line: continue
            obj = json.loads(line)
            qid = obj.get("qid") or obj.get("id") or f"q{n_in+1}"
            q   = obj.get("question") or obj.get("text") or ""
            if not q: continue
            n_in += 1

            rel = detect_relation(q)
            head_cui, tail_cui = choose_head_tail(q, rel, umls_entries)

            # backfill relation from entity types if needed
            if not rel:
                if head_cui.startswith("drug_") and tail_cui.startswith("drug_"):
                    rel = "INTERACTS_WITH"
                elif head_cui.startswith("drug_") and tail_cui.startswith("disease_"):
                    rel = "ADVERSE_EFFECT"
                else:
                    rel = "INTERACTS_WITH"

            # placeholder tail for AE to allow neighbor enumeration downstream (optional)
            if rel == "ADVERSE_EFFECT" and not tail_cui:
                tail_cui = "disease_adverse_effects"

            intent = detect_intent(q)
            out = {
                "qid": qid,
                "question": q,
                "text": q,
                "intent": intent,
                "relations": [rel],
                "head_cui": head_cui,
                "tail_cui": tail_cui
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Analyzed {n_in} raw lines -> {n_out} structured lines: {args.out}")

if __name__ == "__main__":
    main()
