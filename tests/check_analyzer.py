import json, sys, collections, pathlib

# Adjust if your project path changes
proj = pathlib.Path(r"F:\graph-corag-clean")
analyzed = proj / "data" / "queries.analyzed.jsonl"         # analyzer output
gold     = proj / "tests" / "gold.analyzer.40.jsonl"        # your gold labels

def read_jsonl(p):
    with open(p, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Bad JSON at {p}:{ln}\n{line}\n{e}") from e
# index (1..N) → records (assumes same ordering as your current files)
pred = {i+1: obj for i, obj in enumerate(read_jsonl(analyzed))}
gold = {i+1: obj for i, obj in enumerate(read_jsonl(gold))}

def safe_rel(x):
    r = x.get("relations") or []
    return r[0] if r else None

def acc(d):
    return 100.0*d["correct"]/max(1,d["total"])

stats = {
    "intent": {"correct":0,"total":0},
    "relation": {"correct":0,"total":0},
    "head_cui_coverage": {"have":0,"total":0}
}

miss = []

for idx, g in gold.items():
    p = pred.get(idx, {})

    # intent
    stats["intent"]["total"] += 1
    if p.get("intent") == g.get("intent"):
        stats["intent"]["correct"] += 1
    else:
        miss.append((idx, "intent", g.get("intent"), p.get("intent")))

    # relation (just check the first relation label for now)
    stats["relation"]["total"] += 1
    if safe_rel(p) == g.get("relation"):
        stats["relation"]["correct"] += 1
    else:
        miss.append((idx, "relation", g.get("relation"), safe_rel(p)))

    # head_cui coverage
    stats["head_cui_coverage"]["total"] += 1
    if p.get("head_cui"):
        stats["head_cui_coverage"]["have"] += 1
    else:
        miss.append((idx, "head_cui", "non-empty", p.get("head_cui")))

print("\n=== Analyzer sanity report ===")
print(f"Intent accuracy   : {acc(stats['intent']):5.1f}%  ({stats['intent']['correct']}/{stats['intent']['total']})")
print(f"Relation accuracy : {acc(stats['relation']):5.1f}%  ({stats['relation']['correct']}/{stats['relation']['total']})")
cov = 100.0*stats["head_cui_coverage"]["have"]/max(1,stats["head_cui_coverage"]["total"])
print(f"head_cui coverage : {cov:5.1f}%  ({stats['head_cui_coverage']['have']}/{stats['head_cui_coverage']['total']})")

if miss:
    print("\n--- Mismatches / Gaps ---")
    for idx, kind, exp, got in miss:
        print(f"#{idx:02d} {kind:8}  expected={exp}  got={got}")
else:
    print("\nNo mismatches. ✔")
