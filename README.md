# TDA-multiscale

Multi-scale intrinsic dimension of text: an extension of the qPHD method to ModernBERT
embeddings. The estimator is recalibrated, a controlled intervention study establishes
which properties of a text determine the dimension it measures, and the measurement is
then traced down to the geometry of the embedding cloud and turned into a detector for
damaged documents on the COLING 2025 corpus.

**Start here:**

- [EXPERIMENTS.md](EXPERIMENTS.md) — **every stage of the work in one place**: what was
  measured, what was found, and which document holds the detail. Start here if you are
  picking the project up.
- [phdq_ext/OVERVIEW.md](phdq_ext/OVERVIEW.md) — every figure and experiment in one table
- [phdq_ext/report/qphd_report.pdf](phdq_ext/report/qphd_report.pdf) — the write-up, 7 pages
- [phdq_ext/README.md](phdq_ext/README.md) — method, formulas and findings in detail (Russian)

## Headline results

**The estimator** (stage 1, FLAT dataset)

- The published edge-counting rule, `floor(q·m)`, produces a saw-tooth artefact in d(q)
  that is invisible on a q grid of step 0.1. Fixed by fractional boundary weighting on an
  aligned grid.
- Relative noise of a single measurement falls from 0.114 to 0.026 after recalibration.
- d ∝ L^−0.21 in the number of sampled tokens and does not converge within reach, so L
  must be matched exactly and absolute dimensions are properties of a measurement window.
- Across 3000 texts, every generator sits above human text at every q. Separability peaks
  near q = 0.3, i.e. the trimmed estimator beats the untrimmed one.
- Four causal factors, from 31 perturbations: lexical diversity, word order within a
  sentence, topic variety, idea count. Sentence length is not among them.
- Two methodological findings: every LLM rewrite carries a shared component that raises
  dimension regardless of the instruction, and a weakly executed instruction yields a
  result with the wrong sign rather than a null one.

**Three bands as a defect detector** (stage 2, COLING 2025)

- The sign of a band matters and its magnitude does not: a band falling below the corpus
  norm means a damaged document, a band rising means nothing. The `−−−` profile is right
  64% of the time against a 14% base rate; any rise is right 5% of the time, below base.
- Two thirds of the human texts in COLING carry a detectable collection artefact, so every
  contrast measured on that corpus is compressed. `reddit_eli5` is detokenised wholesale.
- A subcorpus-wide defect shows as a shift of the subcorpus, not as outliers inside it,
  which is why normalising by genre subtracts the very thing being looked for.

**What the bands measure** (stage 3, geometry)

- The cloud is a thin shell on which each token type owns a cell. Whether a token has a
  second copy in the sample decides almost everything about the tree: with a twin it is
  its own nearest neighbour 97% of the time, without one it hangs off a stranger.
- Over 27 perturbations, each band answers to a different property: the fine band is the
  radius of a cell (r = +0.91), the middle band is the share of once-used tokens (+0.92),
  the coarse band is the evenness of the long edges (−0.58…−0.74).
- ModernBERT has a second distinguished direction beside the attention sink: the axis a
  token's vector moves along when its context determines nothing. The same axis for two
  unrelated corruptions (cosine +0.743). It separates a broken text from a merely dull one,
  which the bands cannot.
- 2.2% of tokens form an attention sink that inflates every band by 4–15% and is an
  artefact of the encoder, not the text. It should be removed before the estimate.

## Layout

```
EXPERIMENTS.md          every stage of the work, with pointers to the detail
phdq_ext/scripts/       core: estimator, embedder, perturbations, runners, judges
phdq_ext/experiments/   131 experiment scripts; each opens with the question it answers
phdq_ext/results/       per-text results, generated texts, summary tables, the reports
phdq_ext/figures/       figures
coling/                 the COLING 2025 MGT pool (stages 2-3)
data_completion/        the FLAT dataset (stage 1)
```

The long-form reports live in `phdq_ext/results/`: `STATE.md` and `HYPOTHESES.md` for the
detector, `GEOMETRY.md` for the structure of the cloud, `SESSION_2026_09_14.md`,
`PERT_GEOMETRY.md` and `COARSE_BAND_MODEL.md` for what each band measures.

Third-party material: `data_completion/` is the FLAT dataset, `coling/` is the COLING 2025
MGT detection corpus; `phdq_ext/scripts/reference/phd_scale_original.py` is the reference
implementation from
[candelabrum/PHDQ](https://github.com/candelabrum/PHDQ), retained unmodified for
comparison; the PDF at the root is the paper being extended. Rights to these remain with
their authors.

## Running it

```bash
cd phdq_ext
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch transformers scipy scikit-learn pandas matplotlib seaborn tqdm

# stage 1, FLAT: curves per genre and per source
.venv/bin/python scripts/run_qphd.py --n-texts 150

# stages 2-3, COLING: the corpus everything after stage 1 is measured on
.venv/bin/python scripts/run_coling.py --n-texts 120

# any single experiment, from phdq_ext/
.venv/bin/python experiments/edge_taxonomy.py
```

Embeddings are cached to `phdq_ext/cache/` (38 GB at the sizes used here, not committed);
the first run of any script fills it and later runs are fast. `coling/dev.parquet`
(234 MB) is not committed either and is re-downloadable from HuggingFace; the stratified
pool drawn from it is. The LLM rewrites need an OpenRouter key in `phdq_ext/.env`, but
every rewrite used in the work is already saved under `phdq_ext/results/*.json`, so the
generation does not need repeating.
