# TDA-multiscale

Multi-scale intrinsic dimension of text: an extension of the qPHD method to ModernBERT
embeddings, with the estimator calibrated and a controlled intervention study asking
which properties of a text causally determine the dimension it measures.

**Start here:**

- [EXPERIMENTS.md](EXPERIMENTS.md) — **every stage of the work in one place**: what was
  measured, what was found, and which document holds the detail. Start here if you are
  picking the project up.
- [phdq_ext/OVERVIEW.md](phdq_ext/OVERVIEW.md) — every figure and experiment in one table
- [phdq_ext/report/qphd_report.pdf](phdq_ext/report/qphd_report.pdf) — the write-up, 7 pages
- [phdq_ext/README.md](phdq_ext/README.md) — method, formulas and findings in detail (Russian)

## Headline results

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

## Layout

```
phdq_ext/scripts/       core: estimator, embedder, runners, report builder
phdq_ext/experiments/   the parameter sweeps that selected the configuration
phdq_ext/results/       per-text results, generated texts, summary tables
phdq_ext/figures/       figures
data_completion/        the FLAT dataset used throughout
```

Third-party material: `data_completion/` is the FLAT dataset;
`phdq_ext/scripts/reference/phd_scale_original.py` is the reference implementation from
[candelabrum/PHDQ](https://github.com/candelabrum/PHDQ), retained unmodified for
comparison; the PDF at the root is the paper being extended. Rights to these remain with
their authors.

## Running it

```bash
cd phdq_ext
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch transformers scipy scikit-learn pandas matplotlib seaborn tqdm
.venv/bin/python scripts/run_qphd.py --n-texts 150
```

Embeddings are cached to `phdq_ext/cache/` (~2.7 GB, not committed). The LLM rewrites need
an OpenRouter key in `phdq_ext/.env`.
