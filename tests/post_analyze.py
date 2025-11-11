import json, re, sys, io

def read_jsonl(path):
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            yield json.loads(line)

def write_jsonl(path, objs):
    with io.open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False)+"\n")

# Intent heuristic:
# - explicit chain cue -> multihop
# - yes/no if starts with Does/Is/Are/Can/Do/Should
# - else factoid
YN_PREFIX = re.compile(r"^(does|is|are|can|do|should)\b", re.I)
CHAIN_CUE = re.compile(r"\bif .* and .*[,;]", re.I)

def infer_intent(text, relations):
    if CHAIN_CUE.search(text):
        return "multihop"
    if YN_PREFIX.search(text):
        return "yesno"
    return "factoid"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: post_analyze.py <enriched.jsonl> <out.analyzed.jsonl>", file=sys.stderr)
        sys.exit(2)
    inp, outp = sys.argv[1], sys.argv[2]
    out = []
    for obj in read_jsonl(inp):
        text = obj.get("text","")
        rels = obj.get("relations") or []
        obj["intent"] = infer_intent(text, rels)
        out.append(obj)
    write_jsonl(outp, out)
