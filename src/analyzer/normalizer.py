import re, unicodedata, yaml
from pathlib import Path

class NameNormalizer:
    def __init__(self, cfg_path):
        self.cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        self._prefix = [re.compile(p, flags=re.I) for p in self.cfg.get("prefix_strippers", [])]

    def _base(self, s:str)->str:
        if self.cfg.get("unicode_nfkc", True):
            s = unicodedata.normalize("NFKC", s)
        if self.cfg.get("lowercase", True):
            s = s.lower()
        if self.cfg.get("dedupe_whitespace", True):
            s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _variants(self, s:str):
        outs = [s]
        for rule in self.cfg.get("variants", []):
            if "replace" in rule and isinstance(rule["replace"], (list,tuple)) and len(rule["replace"])==2:
                a,b = rule["replace"]
                outs = [x.replace(a,b) for x in outs]
            if rule.get("collapse_spaces"):
                outs = [re.sub(r"\s+"," ", x) for x in outs]
        return outs

    def normalize_list(self, s:str)->list[str]:
        if not isinstance(s, str) or not s.strip():
            return []
        raw = s.strip()
        # strip prefixes
        s2 = raw
        for rx in self._prefix:
            s2 = rx.sub("", s2)
        base = self._base(s2)
        vars_ = [base] + self._variants(base) + [base.title()]
        # unique, case-insensitive
        seen, out = set(), []
        for x in [raw] + vars_:
            k = x.lower()
            if k and k not in seen:
                seen.add(k); out.append(x)
        return out
