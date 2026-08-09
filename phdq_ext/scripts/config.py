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
  aligned grid    every m = n-1 a multiple of 20, so that q*m is an integer for
                  the whole q grid and no rounding arises in the first place.
                  Grid *shape* turned out not to matter at all (all variants
                  within 0.10-0.115), so nothing is lost by snapping the
                  log-spaced targets to multiples of 20. Grid *range* does
                  matter: extending below 0.2L pulls in the small-n regime and
                  inflates d, hence L is chosen so the range stays at 5x.

L is set as an absolute token count, not as a fraction of the text: because
d ~ L^-0.2, a fractional L makes d a function of text length (regression
slope -0.27..-0.56 vs -0.01..-0.07 for absolute L; exp_length_policy.py).
L is also kept independent of q: scaling it as N/(1-q) does equalise the
retained edge count, but the noise it was meant to cure is already gone in
this config, and the rescaling bends the d(q) curve by the known
L-dependence instead of adding information (exp_L_per_q.py).

Fractional trimming (qphd.trimmed_sum, trim="fractional") is on by default and
is independent of this config; trim="floor" reproduces the paper. trim="round"
exists only to document that rounding is not a fix: it removes a
floating-point edge case but leaves the artefact itself (exp_grid_alignment.py).
"""
import numpy as np

# L = 201 gives the aligned grid n = 41, 61, 81, 101, 121, 161, 201, whose edge
# counts m = 40..200 span exactly the 5x range everything else was validated on,
# and it keeps the most texts: 1933 across the four genres, of which 460
# academic_abstracts (against 297 at L = 256 and 186 at L = 301). Noise is 23%
# higher than at L = 261, but a genre mean gains precision as sqrt(N), so the
# 1.64x larger sample for the weakest genre more than compensates.
L_DEFAULT = 201
POOL_FACTOR = 2  # pool size = POOL_FACTOR * L
REPLICATES = 32
REPLACE = False
Q_STEP = 0.05
Q_GRID = tuple(np.round(np.arange(0, 0.901, Q_STEP), 2))
ALIGN_STEP = int(round(1 / Q_STEP))  # every m = n-1 a multiple of this
N_POINTS = 8
F_MIN = 0.2  # smallest subsample as a fraction of the largest
LOG8 = tuple(np.round(np.logspace(np.log10(F_MIN), 0.0, N_POINTS), 3))


def aligned_grid(L, n_points=N_POINTS, f_min=F_MIN, step=ALIGN_STEP, min_points=6):
    """Subsample sizes n whose edge counts m = n-1 are multiples of `step`.

    With a q grid of step 1/`step`, every q*m is then an integer and no
    rounding is needed at all. Points are log-spaced over [f_min*m_max, m_max]
    and snapped to the nearest multiple of `step`.

    Returns None when L is too small for the alignment to leave enough
    distinct points: multiples of `step` are coarse at the low end, so below
    L ~ 200 the grid collapses and log fractions are the better choice.
    """
    m_max = (L - 1) // step * step
    if m_max < 2 * step:
        return None
    m_min = max(step, int(round(f_min * m_max / step)) * step)
    targets = np.logspace(np.log10(m_min), np.log10(m_max), n_points)
    m = np.unique((np.round(targets / step) * step).astype(int))
    if m.size < min_points:
        return None
    return tuple((m + 1).tolist())

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


def qphd_kwargs(L=L_DEFAULT, pool=None, **overrides):
    """Standard keyword arguments for qphd().

    Uses the aligned grid when L allows it and falls back to log-spaced
    fractions otherwise; the fallback matters only for the L sweeps.
    """
    kw = dict(replicates=REPLICATES, replace=REPLACE, pool=pool)
    grid = aligned_grid(L)
    if grid is None:
        kw["n_fraction_list"] = LOG8
    else:
        kw["n_values"] = grid
    kw.update(overrides)
    return kw
