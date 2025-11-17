from src.analyzer.entity_linking_adapter import ELAdapter
from src.analyzer.sapbert_linker_typeaware import TypeAwareLinker
from src.analyzer.reranker_simple import ContextReranker

linker = TypeAwareLinker(
    model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    index_root=r"F:\graph-corag-clean\artifacts\concept_index",
    norm_cfg=r"F:\graph-corag-clean\config\name_normalization.yaml",
)
rerank = ContextReranker()  # optional

adapter = ELAdapter(linker=linker, reranker=rerank, default_topk=8)

cases = [
    ("Does adalimumab interact with CD274 protein?", ["adalimumab"], "INTERACTS_WITH", "head"),
    ("Which drugs interact with CTLA4 protein?", ["CTLA4 protein"], "INTERACTS_WITH", "tail"),
    ("What adverse effect is associated with etanercept?", ["etanercept"], "ADVERSE_EFFECT", "head"),
]

for q, m, rel, slot in cases:
    cands = adapter.link_mentions(q, m, rel, slot, topk=5)[0]
    print(rel, slot, "->", [ (c["kg_id"], c["entity_type"], round(c["score"],3)) for c in cands[:3] ])
