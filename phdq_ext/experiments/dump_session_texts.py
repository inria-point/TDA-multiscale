"""Сохранить тексты механических пертурбаций этой сессии, как они были поданы.

Переписывания моделью лежат в results/*.json с самого их создания. Механические
пертурбации geometry-прогонов создавались на лету из зерна — воспроизводимо, но
нечитаемо: чтобы увидеть, что именно подавалось на вход, приходилось запускать
код. Здесь они выгружаются в том же формате, что и остальные:

    {пертурбация: {ключ текста: текст}}

Зёрна те же, что в прогонах (seed = индекс текста), так что содержимое
совпадает с тем, на чём считались полосы.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from pert_geometry import PERTS as MECH
from perturb import apply as perturb

BASE = os.path.join(HERE, "..")
N_TEXTS = 50


def main():
    pool = pd.read_parquet(os.path.join(BASE, "..", "coling", "pool.parquet"))
    hum = [(str(r.id), r.text) for r in pool.itertuples() if r.is_human]
    out = {}
    for label, name in MECH:
        d = {}
        for i, (key, s) in enumerate(hum[:N_TEXTS]):
            try:
                d[key] = s if name == "identity" else perturb(name, s, seed=i)
            except Exception as exc:                      # noqa: BLE001
                print(f"  {name}: {exc}", flush=True)
                break
        if d:
            out[name] = d
            print(f"{label:26s} ({name}) — {len(d)} текстов", flush=True)
    p = os.path.join(BASE, "results", "perturbed_texts_geometry.json")
    with open(p, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n{p}: {len(out)} пертурбаций, "
          f"{os.path.getsize(p) / 1048576:.1f} МБ")


if __name__ == "__main__":
    main()
