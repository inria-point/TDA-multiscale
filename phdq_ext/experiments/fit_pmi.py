"""Is "local appropriateness" measurable, and is it what the long edges read?

HYPOTHESES.md 3a says the coarse band reacts to rare tokens and that their
*local* appropriateness sets the sign: a rare word the surroundings put there
meaningfully lifts the band, a rare token nothing puts there lowers it. That is
a statement about text with no geometric content, and the cell model has no
coordinate for it -- GEOMETRY.md 5 leaves the coarse band at 29% of what the
noise floor allows.

Appropriateness has an obvious measurement: the encoder itself. Mask the token,
read the probability the model gives it back, and subtract its base rate in the
corpus. What is left is pointwise mutual information -- how much this context
raises the odds of this word over meeting it at random:

    PMI = log p(token | context) - log p(token)

High PMI is a word the surroundings call for. PMI near zero is a word that
could have been anywhere: the mechanically substituted foreign word, the
tokenisation artefact. Rarity itself is in the second term and divided out.

Two questions, one script:

  1. Across singleton tokens of ordinary text, does PMI predict how far the
     token hangs from the tree -- the petiole the coarse band is made of?
  2. hapax_swap_wide replaces singletons with foreign words and drops the
     coarse band 19% with every lexical statistic fixed. Does PMI see the
     substitution, and does the geometry of those tokens move with it?

The geometric half needs a coordinate of its own, and there is a natural one.
Split a token's vector into where its type usually sits and how far this
occurrence was carried from there:

    дом(тип)   the mean vector of that token over the corpus
    смещение   ||v - дом(тип)|| , in units of the text's mean pairwise distance

The prediction the project already holds says these go in *opposite*
directions. Shuffling makes every context meaningless at once, and the cells of
content words swell 17% (shuffle_geometry.py): an occurrence the surroundings do
not call for is pushed away from where its type lives. So low PMI should mean a
*larger* смещение, not a smaller one, and a mechanically substituted word should
sit further from its home than the word it replaced.

That is a prediction about a token with a cell to be pushed out of. A singleton
has none, and what happens to it is a separate question -- the first measurement
here says its petiole gets *shorter*, so whatever direction it is pushed in, it
is not away from the rest of the text. Two readings, and the numbers decide:

  a. отталкивание       the vector leaves its home in some text-specific
                        direction, and lands wherever
  b. общее состояние    the encoder has nothing to encode, falls back on a
                        generic representation, and all such tokens land in the
                        same place -- near the centre of the cloud and near each
                        other

They differ in a measurable way: under (b) the substituted tokens should sit
closer to the text centroid and closer to one another than the words they
replaced, under (a) neither.

One more distinction the shuffle result does not settle. A cell swells either
because the occurrences are carried *further* from home, or because they are
carried the same distance in *disagreeing* directions. Both are measured here:
смещение is the magnitude, согласие is the cosine between one occurrence's
displacement and the mean displacement of its type. Shuffling should destroy
согласие; whether it also raises смещение is open.
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from transformers import AutoModelForMaskedLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP, corpus_ranks, token_class
from embedder import MODEL_NAME, Embedder
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
N_TEXTS = int(os.environ.get("N_TEXTS", 40))
BATCH = int(os.environ.get("BATCH", 16))
CONDS = [("исходный", None), ("чужие хапаксы", "hapax_swap_wide"),
         ("перемешивание", "shuffle_words")]


HOME_TEXTS = int(os.environ.get("HOME_TEXTS", 200))
HOME_MIN = 5


def home_vectors(texts, emb):
    """Mean vector of every token type over the corpus: its lexical home."""
    tot, cnt = {}, Counter()
    for i, s in enumerate(texts):
        e = emb.embed_cached(s, cache_key=f"home_{i}")
        toks = emb.tokenizer.tokenize(s)
        if len(toks) != e.shape[0]:
            continue
        for t, vv in zip(toks, e):
            if t in tot:
                tot[t] += vv
            else:
                tot[t] = vv.copy()
            cnt[t] += 1
    return {t: tot[t] / cnt[t] for t in tot if cnt[t] >= HOME_MIN}


def unigram(texts, emb):
    c = Counter()
    for s in texts:
        c.update(emb.tokenizer.tokenize(s))
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}, n


@torch.no_grad()
def pmi_of(text, emb, mlm, logp0, device):
    """PMI of every token of the text, one masked forward per position."""
    enc = emb.tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=emb.max_length)
    ids = enc["input_ids"][0].to(device)
    special = torch.tensor(emb.tokenizer.get_special_tokens_mask(
        ids.tolist(), already_has_special_tokens=True), dtype=torch.bool)
    pos = torch.arange(len(ids))[~special]
    toks = emb.tokenizer.convert_ids_to_tokens(ids[~special].tolist())
    out = np.zeros(len(pos))
    mask_id = emb.tokenizer.mask_token_id
    for s in range(0, len(pos), BATCH):
        chunk = pos[s:s + BATCH]
        batch = ids.unsqueeze(0).repeat(len(chunk), 1).clone()
        batch[torch.arange(len(chunk)), chunk] = mask_id
        att = torch.ones_like(batch)
        lg = mlm(input_ids=batch, attention_mask=att).logits
        lp = torch.log_softmax(lg[torch.arange(len(chunk)), chunk].float(), -1)
        out[s:s + len(chunk)] = lp[torch.arange(len(chunk)),
                                   ids[chunk]].cpu().numpy()
    base = np.array([logp0.get(t, logp0["__unk__"]) for t in toks])
    return toks, out, out - base


def tree_stats(v):
    D = squareform(pdist(v))
    scale = pdist(v).mean()
    mst = minimum_spanning_tree(D).tocoo()
    deg = np.zeros(len(v), int)
    longest = np.zeros(len(v))
    for a, b, w in zip(mst.row, mst.col, mst.data / scale):
        deg[a] += 1
        deg[b] += 1
        longest[a] = max(longest[a], w)
        longest[b] = max(longest[b], w)
    return deg, longest, np.percentile(mst.data / scale, 80)


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    emb = Embedder()
    device = emb.device
    mlm = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    ranks = corpus_ranks(hum, emb)
    uni, n_tok = unigram(hum, emb)
    logp0 = {k: float(np.log(v)) for k, v in uni.items()}
    logp0["__unk__"] = float(np.log(0.5 / n_tok))
    print(f"униграммы по {len(hum)} текстам, {n_tok} токенов", flush=True)
    home = home_vectors(hum[:HOME_TEXTS], emb)
    print(f"домашних векторов: {len(home)} типов "
          f"(>= {HOME_MIN} вхождений в {HOME_TEXTS} текстах)", flush=True)

    rows = []
    for i, s in enumerate(hum[:N_TEXTS]):
        for label, name in CONDS:
            txt = s if name is None else perturb(name, s, seed=i)
            toks, lp, pmi = pmi_of(txt, emb, mlm, logp0, device)
            e = emb.embed_cached(txt, cache_key=f"pg_{name or 'identity'}_{i}")
            keep = [j for j, t in enumerate(toks) if t not in SKIP]
            if len(keep) < cfg.L_DEFAULT or e.shape[0] != len(toks):
                continue
            e, toks = e[keep], [toks[j] for j in keep]
            lp, pmi = lp[keep], pmi[keep]
            rng = np.random.default_rng(1000)
            idx = rng.choice(len(toks), size=cfg.L_DEFAULT, replace=False)
            v, t = e[idx], [toks[j] for j in idx]
            cnt = Counter(t)
            deg, longest, cut = tree_stats(v)
            scale = pdist(v).mean()
            centre = v.mean(0)
            d_centre = np.linalg.norm(v - centre, axis=1) / scale
            # displacement from the lexical home, and whether the occurrences
            # of one type are displaced in agreeing directions
            disp = {j: v[j] - home[tok] for j, tok in enumerate(t)
                    if tok in home}
            agree = {}
            for tok, occ in Counter(t).items():
                ix = [j for j, x in enumerate(t) if x == tok and j in disp]
                if occ < 2 or len(ix) < 2:
                    continue
                M = np.array([disp[j] for j in ix])
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
                C = M @ M.T
                np.fill_diagonal(C, np.nan)
                for pos, j in enumerate(ix):
                    agree[j] = float(np.nanmean(C[pos]))
            for j, tok in enumerate(t):
                rows.append({
                    "согласие": agree.get(j, np.nan),
                    "текст": i, "условие": label, "токен": tok,
                    "класс": token_class(tok, ranks),
                    "k": cnt[tok], "logp": lp[idx[j]], "PMI": pmi[idx[j]],
                    "смещение": (np.linalg.norm(v[j] - home[tok]) / scale
                                 if tok in home else np.nan),
                    "cos дом": (float(v[j] @ home[tok]
                                      / (np.linalg.norm(v[j])
                                         * np.linalg.norm(home[tok])))
                                if tok in home else np.nan),
                    "до центра": float(d_centre[j]),
                    "черешок": longest[j], "лист": int(deg[j] == 1),
                    "длинный лист": int(deg[j] == 1 and longest[j] > cut)})
        if (i + 1) % 5 == 0:
            print(f"  {i + 1} текстов, {len(rows)} токенов", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(BASE, "results", "fit_pmi.csv"), index=False)
    print(f"\nresults/fit_pmi.csv: {len(D)} токенов")


if __name__ == "__main__":
    main()
