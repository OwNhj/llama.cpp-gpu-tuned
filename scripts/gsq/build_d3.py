#!/usr/bin/env python3
"""Build the D3 calibration corpus + paired eval sets from downloaded sources.

Pool -> deterministic md5-hash bucket split 90/10 (calib/eval) per source:
any subset taken from calib has ZERO string overlap with eval sets.
Calibration body = round-robin interleave of source blocks (64KB each) so ANY
contiguous window (e.g. the first 24 chunks) sees all sources ~evenly.
Composition targets (real assistant+code workload):
  en-instruct 25%, zh-instruct 25%, code(magicoder) 25%, en-prose(wikitext-train) 12.5%,
  en-prose(War&Peace, non-eval) 12.5%, zh-prose(wikipedia-zh) 待定(>=10% 若下载成功，
  否则份额并入 en 两栏)
"""
import os, csv, json, hashlib, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))

C = os.environ.get("GSQ_CORPUS", os.path.join(HERE, "corpus"))
OUT = os.environ.get("GSQ_DIR", HERE)
csv.field_size_limit(10**8)
BUCKET_CALIB = 90  # hash%100 < 90 -> calib


def bucket(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % 100


def load_csv_texts(path, fields):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parts = [row.get(k, "") or "" for k in fields]
            t = "\n".join(p for p in parts if p.strip())
            if len(t) >= 200:
                out.append(t)
    return out


def load_jsonlines(path, key_user="input", key_asst="output"):
    out = []
    txt = open(path, encoding="utf-8").read()
    try:
        recs = json.loads(txt)
    except Exception:
        recs = []
        for l in txt.splitlines():
            try:
                recs.append(json.loads(l))
            except Exception:
                continue
    if not isinstance(recs, list):
        recs = [recs]
    for r in recs:
        if not isinstance(r, dict):
            continue
        u = r.get(key_user) or (r.get("messages") or [{}])[0].get("content", "")
        a = r.get(key_asst) or (r.get("messages") or [{}])[-1].get("content", "")
        t = (str(u) + "\n" + str(a)).strip()
        if len(t) >= 200:
            out.append(t)
    return out


def load_txt_paragraphs(path, minlen=400, skip_bytes=0):
    txt = open(path, encoding="utf-8", errors="ignore").read()[skip_bytes:]
    parts = [p.strip() for p in txt.replace("\r", "").split("\n") if p.strip()]
    out, cur = [], ""
    for p in parts:
        cur = (cur + "\n" + p).strip() if cur else p
        if len(cur) >= max(minlen, 2000):
            out.append(cur); cur = ""
    if len(cur) >= minlen:
        out.append(cur)
    return out


def split_pool(items, want):
    calib = [t for t in items if bucket(t) < BUCKET_CALIB]
    ev = [t for t in items if bucket(t) >= BUCKET_CALIB]
    rng = random.Random(7)
    rng.shuffle(calib)
    rng.shuffle(ev)
    return calib[:want], ev


def blocks(texts, size=4096):
    buf = "\n\n".join(texts)
    return [buf[i:i + size] for i in range(0, len(buf), size)]


def interleave(pool_lists, size=4096):
    """4KB-block round-robin across sources; window >= 24KB sees every source."""
    bs = [blocks(p, size) for p in pool_lists]
    out = []
    nmax = max((len(x) for x in bs), default=0)
    for i in range(nmax):
        for lst in bs:
            if i < len(lst):
                out.append(lst[i])
    return "\n\n".join(out)


def main():
    meta = {}
    pools = {}
    en = load_csv_texts(f"{C}/en.csv", ["instruction", "input", "output"])
    pools["en_inst"] = en
    zh = load_csv_texts(f"{C}/zh.csv", ["instruction", "input", "output"])
    pools["zh_inst"] = zh
    mg = load_jsonlines(f"{C}/magicoder.json")
    pools["code"] = mg
    wt = load_txt_paragraphs(f"{C}/wikitext-2-raw/wiki.train.raw")
    pools["en_wiki"] = wt
    # corpus.txt eval uses first 8*4096 tokens (~<600KB); calib uses only bytes AFTER 600K
    pools["prose_en"] = load_txt_paragraphs(os.environ.get("GSQ_PROSE_EN", "corpus.txt"),
                                            minlen=400, skip_bytes=600_000)
    wzp = f"{C}/wzh.json"
    if os.path.exists(wzp) and os.path.getsize(wzp) > 100_000_000:
        try:
            pools["zh_wiki"] = load_jsonlines(wzp, "title", "text")[:20000]
        except Exception as e:
            print("zh_wiki load failed:", e)
    targets = {"en_inst": 0.25, "zh_inst": 0.25, "code": 0.25,
               "en_wiki": 0.125, "prose_en": 0.125}
    if "zh_wiki" in pools:
        targets = {"en_inst": 0.20, "zh_inst": 0.22, "code": 0.22,
                   "en_wiki": 0.10, "prose_en": 0.10, "zh_wiki": 0.16}
    MB = 24 * 512 * 4 * 4  # 24 chunks * 512 tok * ~4 chars/token = ~19.7MB
    calib_parts, eval_parts = {}, {}
    for k, fr in targets.items():
        want = int(MB * fr / 0.85)  # 15% headroom for join noise
        cpart, epart = split_pool(pools[k], want)
        calib_parts[k] = cpart
        eval_parts[k] = epart[:6000]
        meta[k] = dict(pool=len(pools[k]), calib=len(cpart), calib_chars=sum(map(len, cpart)),
                       eval_n=len(eval_parts[k]))
    calib = interleave([calib_parts[k] for k in targets])
    open(f"{OUT}/calib-d3.txt", "w", encoding="utf-8").write(calib)
    # eval sets
    def join(lst): return "\n\n".join(lst)
    eg = eval_parts["en_inst"][:1500] + eval_parts["zh_inst"][:1500] \
         + eval_parts["en_wiki"][:900] + eval_parts["prose_en"][:600]
    open(f"{OUT}/eval-gen.txt", "w", encoding="utf-8").write(interleave([eg]))
    open(f"{OUT}/eval-code.txt", "w", encoding="utf-8").write(join(eval_parts["code"]))
    ez = eval_parts["zh_inst"][:2500] + eval_parts.get("zh_wiki", [])[:2500]
    open(f"{OUT}/eval-zh.txt", "w", encoding="utf-8").write(interleave([ez]))
    open(f"{OUT}/eval-wiki.txt", "w", encoding="utf-8").write(join(eval_parts["en_wiki"]))
    # leak check
    def leak(a, b):
        A = set(t[:200] for t in a)
        B = [t[:200] for t in b]
        return sum(1 for t in B if t in A)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print("leaks(calib vs eval-gen):",
          leak(calib_parts["en_inst"], eval_parts["en_inst"]),
          leak(calib_parts["zh_inst"], eval_parts["zh_inst"]),
          leak(calib_parts["en_wiki"], eval_parts["en_wiki"]))
    print("calib-d3 size:", len(calib))
    # window composition proof: first 2MB of calib
    head = calib[:80 * 1024]
    import re
    cjk = len(re.findall(r"[\u4e00-\u9fff]", head))
    codey = head.count("def ") + head.count("import ") + head.count("```") \
            + head.count(" ()") + head.count("return") + head.count(":\n")
    has = {"en": "the" in head.lower(), "code": codey > 30, "wiki": "wiki" in head.lower() or "==" in head,
           "lit": any(x in head for x in ("said", "Prince", "count")),
           "zh": cjk > 3000, "zhw": "==" not in head and cjk > 100}
    print("window80KB: cjk=%.1f%% code_markers=%d flags=%s" % (100 * cjk / len(head), codey, has))
    assert 0.08 < cjk / len(head) < 0.45 and codey > 30, "window not mixed - abort"


main()
