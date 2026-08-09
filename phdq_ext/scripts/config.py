"""The qPHD estimator configuration settled on after the variant sweep.

Chosen in scripts/variant_sweep.py; results/variant_sweep_L256.csv.
Relative measurement noise cv_within = 0.026 vs 0.114 for the paper's
settings, with no change in what is measured beyond the bootstrap bias.

  replace=False   sampling with replacement leaves ~37% duplicate points at
                  n=L, which distorts the edge-length distribution and biases
                  d upward by 12-18%. Duplicates do not create zero-length
                  edges (scipy reads a zero weight as a missing edge), they
                  create extra short ones.
  pool=2L         drawing each subsample from a wider pool than the L-token
                  subset halves the noise and leaves d unchanged (7.88 vs
                  7.89). A pool of the whole text is quieter still
                  (cv 0.023 vs 0.038) but lets text length leak back in at
                  the extreme tail (slope vs log n_tokens -0.195 at
                  q_large=0.9, against -0.064 for a 2L pool), so 2L is the
                  default: length-independence matters more here than the
                  last few percent of noise. See exp_length_policy.py.
  replicates=32   noise falls 0.114 -> 0.076; 64 gives only 0.066 for twice
                  the compute.
  LOG8            grid *shape* turned out not to matter at all (all variants
                  within 0.10-0.115). Grid *range* matters a lot: extending
                  below 0.2L pulls in the small-n regime and inflates d.

L is set as an absolute token count, not as a fraction of the text: because
d ~ L^-0.2, a fractional L makes d a function of text length (regression
slope -0.27..-0.56 vs -0.01..-0.07 for absolute L; exp_length_policy.py).
L is also kept independent of q: scaling it as N/(1-q) does equalise the
retained edge count, but the noise it was meant to cure is already gone in
this config, and the rescaling bends the d(q) curve by the known
L-dependence instead of adding information (exp_L_per_q.py).

Fractional trimming (qphd.trimmed_sum, discrete=False) is on by default and
is independent of this config; discrete=True reproduces the paper.
"""
import numpy as np

L_DEFAULT = 256
POOL_FACTOR = 2  # pool size = POOL_FACTOR * L
REPLICATES = 32
REPLACE = False
LOG8 = tuple(np.round(np.logspace(np.log10(0.2), 0.0, 8), 3))
Q_GRID = tuple(np.round(np.arange(0, 0.901, 0.05), 2))

# the paper's settings, kept for reproduction
LEGACY = dict(
    n_fraction_list=(0.2, 0.4, 0.6, 0.8, 1.0),
    replicates=8,
    replace=True,
    pool=None,
)


def make_pool(embeds, L, rng, factor=POOL_FACTOR):
    """Sample the pool the subsamples are drawn from: factor*L tokens.

    Fixing the pool size across texts is what keeps d free of text length.
    """
    size = min(factor * L, embeds.shape[0])
    return embeds[rng.choice(embeds.shape[0], size, replace=False)]


def qphd_kwargs(pool=None, **overrides):
    """Standard keyword arguments for qphd()."""
    kw = dict(
        n_fraction_list=LOG8,
        replicates=REPLICATES,
        replace=REPLACE,
        pool=pool,
    )
    kw.update(overrides)
    return kw
