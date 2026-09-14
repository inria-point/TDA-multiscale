"""Что это за ось: «исходное» направление или «бессмысленное»?

Ось построена как средний снос вектора при перемешивании (fit_pmi_axis.npz), то
есть по определению указывает от осмысленного состояния к бессмысленному:
проекция растёт с −0.006 в целом тексте до +0.31 в перемешанном. Но это
определение, а не опознание.

Опознание возможно прямое. У ModernBERT входная и выходная матрицы эмбеддингов
связаны (tie_word_embeddings=True), и логит токена есть скалярное произведение
представления с его выходным вектором. Значит сдвиг скрытого состояния вдоль u
поднимает логиты тех токенов, на которые u смотрит. Если u — это «контекст
ничего не диктует», сверху должны стоять самые общие, предсказуемые без
контекста токены.

Считается двумя способами:

  сырой      косинус u со строками матрицы эмбеддингов
  через голову  ModernBERT ставит между скрытым состоянием и W_out блок
                head.dense + head.norm, и мерить надо после него (это записано
                в vocabulary-geometry-baseline). Голова нелинейна, поэтому
                сдвиг оценивается численно: logits(h + eps*u) − logits(h) на
                реальных скрытых состояниях

Плюс: у каких токенов собственный снос в НЕИСПОРЧЕННОМ тексте уже лежит вдоль
оси — то есть кто живёт в этом состоянии всегда.
"""
import hashlib
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import config as cfg
from edge_taxonomy import SKIP
from embedder import CACHE_DIR, MODEL_NAME

BASE = os.path.join(HERE, "..")
HOME_TEXTS, HOME_MIN, NORM_CUT = 200, 5, 30.0
TOPK = 25


def cached(text, key):
    d = hashlib.md5((MODEL_NAME + "\x00" + text).encode()).hexdigest()
    p = os.path.join(CACHE_DIR, f"{key}_{d[:10]}.npz")
    return np.load(p)["embeds"] if os.path.exists(p) else None


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [r.text for r in pool.itertuples() if r.is_human]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    Z = np.load(os.path.join(BASE, "results", "fit_pmi_axis.npz"))
    u = Z["перемешивание"] / np.linalg.norm(Z["перемешивание"])

    mdl = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to("cpu").eval()
    W = mdl.get_input_embeddings().weight.detach()
    Wn = W / W.norm(dim=1, keepdim=True)
    ut = torch.tensor(u, dtype=W.dtype)
    cos = (Wn @ ut).numpy()
    order = np.argsort(cos)[::-1]
    print("1. Косинус оси со строками матрицы эмбеддингов\n")
    print("   ближе всего к оси:")
    print("     " + "  ".join(
        f"{tok.convert_ids_to_tokens(int(i))!r}" for i in order[:TOPK]))
    print("   дальше всего (противоположный конец):")
    print("     " + "  ".join(
        f"{tok.convert_ids_to_tokens(int(i))!r}" for i in order[-TOPK:]))

    # 2. через голову: как сдвиг вдоль оси меняет логиты на реальных состояниях
    h = []
    for i, s in enumerate(hum[:20]):
        e = cached(s, f"home_{i}")
        if e is not None:
            h.append(e[:60])
    H = torch.tensor(np.concatenate(h), dtype=W.dtype)
    with torch.no_grad():
        base = mdl.head(H) @ W.T
        moved = mdl.head(H + 6.0 * ut) @ W.T
        d = (moved - base).mean(0).numpy()
    o2 = np.argsort(d)[::-1]
    print("\n2. Что поднимается в предсказании при сдвиге вдоль оси "
          "(через head, на реальных состояниях)\n")
    print("   растут сильнее всего:")
    print("     " + "  ".join(
        f"{tok.convert_ids_to_tokens(int(i))!r}" for i in o2[:TOPK]))
    print("   падают сильнее всего:")
    print("     " + "  ".join(
        f"{tok.convert_ids_to_tokens(int(i))!r}" for i in o2[-TOPK:]))

    # 3. кто уже живёт в этом состоянии: снос вдоль оси в целом тексте, по типам
    tot, cnt = {}, Counter()
    for i, s in enumerate(hum[:HOME_TEXTS]):
        e, t = cached(s, f"home_{i}"), tok.tokenize(s)
        if e is None or len(t) != e.shape[0]:
            continue
        for x, vv in zip(t, e):
            tot[x] = tot[x] + vv if x in tot else vv.copy()
            cnt[x] += 1
    home = {x: tot[x] / cnt[x] for x in tot if cnt[x] >= HOME_MIN}
    acc = defaultdict(list)
    for i, s in enumerate(hum[:HOME_TEXTS]):
        e, t = cached(s, f"home_{i}"), tok.tokenize(s)
        if e is None or len(t) != e.shape[0]:
            continue
        sc = np.linalg.norm(e, axis=1)
        for x, vv, nm in zip(t, e, sc):
            if x in home and nm >= NORM_CUT:
                acc[x].append(float((vv - home[x]) @ u))
    med = {x: float(np.mean(v)) for x, v in acc.items() if len(v) >= 20}
    S = pd.Series(med).sort_values()
    print("\n3. Чей собственный снос в НЕИСПОРЧЕННОМ тексте уже лежит "
          "вдоль оси\n")
    print("   всегда «бессмысленные» (снос вдоль оси больше всех):")
    print("     " + "  ".join(f"{k!r}" for k in S.index[-TOPK:][::-1]))
    print("   всегда «осмысленные» (снос против оси):")
    print("     " + "  ".join(f"{k!r}" for k in S.index[:TOPK]))
    print(f"\n   медиана по {len(S)} типам: {S.median():+.3f}; "
          f"доля типов со сносом вдоль оси: {(S > 0).mean() * 100:.0f}%")
    S.to_csv(os.path.join(BASE, "results", "axis_decode_tokens.csv"))


if __name__ == "__main__":
    main()
