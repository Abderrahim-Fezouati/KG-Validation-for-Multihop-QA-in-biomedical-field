import json, sys, pathlib

inp  = pathlib.Path(sys.argv[1])
outp = pathlib.Path(sys.argv[2])

def as_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    return [x]

def norm_slot(slot):
    out = []
    for it in as_list(slot):
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict) and "kg_id" in it:
            out.append(it["kg_id"])
    return out

with inp.open("r", encoding="utf-8") as f, outp.open("w", encoding="utf-8") as w:
    for line in f:
        if not line.strip(): continue
        obj = json.loads(line)
        head = obj.get("head") or (obj.get("entities") or {}).get("head") or []
        tail = obj.get("tail") or (obj.get("entities") or {}).get("tail") or []
        out = {
            "qid":       obj.get("qid") or obj.get("id") or "",
            "text":      obj.get("text") or obj.get("q") or obj.get("question") or "",
            "relations": obj.get("relations") or as_list(obj.get("rel") or obj.get("relation")),
            "head":      norm_slot(head),
            "tail":      norm_slot(tail),
        }
        w.write(json.dumps(out, ensure_ascii=False) + "\n")
