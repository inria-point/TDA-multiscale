"""Build the PDF research note from the committed results and figures.

Everything quoted in the text is either read from results/ at build time or
listed here as a constant taken from a committed table, so the report cannot
drift away from the data it describes.
"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

BASE = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(BASE, "figures")
OUT = os.path.join(BASE, "report", "qphd_report.pdf")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold",
                            fontSize=17, leading=21, spaceAfter=2),
    "sub": ParagraphStyle("sub", parent=ss["Normal"], fontName="Times-Italic",
                          fontSize=10.5, leading=14, alignment=TA_CENTER,
                          textColor=colors.HexColor("#444444"), spaceAfter=16),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="Times-Bold",
                         fontSize=12.5, leading=15, spaceBefore=14,
                         spaceAfter=6, textColor=colors.black),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold",
                         fontSize=10.8, leading=13, spaceBefore=9,
                         spaceAfter=4, textColor=colors.black),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman",
                           fontSize=9.9, leading=13.4, alignment=TA_JUSTIFY,
                           spaceAfter=6),
    "abstract": ParagraphStyle("ab", parent=ss["Normal"], fontName="Times-Roman",
                               fontSize=9.4, leading=12.6, alignment=TA_JUSTIFY,
                               leftIndent=0.9 * cm, rightIndent=0.9 * cm,
                               spaceAfter=10),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontName="Times-Roman",
                          fontSize=8.4, leading=11, alignment=TA_JUSTIFY,
                          textColor=colors.HexColor("#333333"),
                          spaceBefore=4, spaceAfter=12),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def H(text, level=1):
    return Paragraph(text, S[f"h{level}"])


def fig(name, caption, width=16.4):
    """Figure scaled to `width` cm, preserving aspect ratio."""
    from reportlab.lib.utils import ImageReader

    path = os.path.join(FIG, name)
    iw, ih = ImageReader(path).getSize()
    w = width * cm
    # bind image to its caption so a page break cannot separate them
    return [KeepTogether([Image(path, width=w, height=w * ih / iw),
                          P(caption, "cap")])]


def table(data, widths=None, align_left_first=True, size=8.3):
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Times-Bold", size),
        ("FONT", (0, 1), (-1, -1), "Times-Roman", size),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
    ]
    if align_left_first:
        style.append(("ALIGN", (0, 0), (0, -1), "LEFT"))
    t.setStyle(TableStyle(style))
    return t


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.5)
    canvas.drawCentredString(A4[0] / 2, 1.15 * cm, str(canvas.getPageNumber()))
    canvas.restoreState()


def story():
    s = []

    s.append(P("Multi-scale intrinsic dimension of text:<br/>"
               "estimator calibration and a controlled intervention study",
               "title"))
    s.append(P("A research note extending the qPHD method to ModernBERT "
               "embeddings", "sub"))

    s.append(P(
        "<b>Abstract.</b> The persistent-homology dimension of a text's token "
        "embedding cloud can be measured at a chosen scale by discarding a "
        "fraction q of the minimum spanning tree's edges before summing them. "
        "We port this estimator to ModernBERT embeddings, calibrate it, and use "
        "it to ask which properties of a text causally determine the dimension "
        "it measures. Calibration proves necessary rather than cosmetic: the "
        "published edge-counting rule introduces a quantisation artefact that "
        "is invisible on a coarse q grid, and the estimator's value depends "
        "strongly on the number of tokens sampled, a dependence no "
        "configuration removes. A reconfigured estimator reduces the relative "
        "noise of a single measurement from 0.114 to 0.026. Applied to 3000 "
        "texts across four genres and five sources, it places every generator "
        "above human text at every q, with separability maximised near q = 0.3 "
        "rather than at q = 0, i.e. above the untrimmed estimator. A controlled "
        "intervention study over 31 perturbations isolates four causal factors: "
        "lexical diversity, word order within a sentence, topic variety and "
        "idea count. Two methodological findings constrain how such studies "
        "must be run: every LLM rewrite carries a shared component that raises "
        "dimension regardless of the instruction, and an instruction that is "
        "only weakly executed yields not a null result but a result of the "
        "wrong sign.", "abstract"))

    # ---------------------------------------------------------------- 1
    s.append(H("1. Method"))
    s.append(P(
        "A text is passed through ModernBERT-base, giving one 768-dimensional "
        "point per token. The intrinsic dimension of that cloud is estimated "
        "from the growth of the total minimum-spanning-tree edge length with "
        "the number of points, S(n) &#8733; n<super>b</super> with b = 1 &#8722; α/d, "
        "so that d = α/(1 &#8722; b), with α = 1 throughout. Discarding a fraction q of the "
        "edges before summing selects a scale: short edges carry local "
        "structure, long edges global structure. Three trimming modes are used "
        "— <i>q_small</i> discards the q shortest edges and so observes the "
        "coarse scale, <i>q_large</i> discards the q longest and observes the "
        "fine scale, and <i>q0.5_range</i> trims from both sides and keeps the "
        "central half, a sliding window."))
    s.append(P(
        "One measurement fixes L tokens, forms a grid of subsample sizes n, "
        "draws 32 subsamples at each n, builds an MST per subsample, trims, "
        "sums, averages, and fits a straight line to log S against log n. The "
        "MST is built once per subsample and reused for every q and mode, since "
        "trimming is only a different sum over one sorted edge list. Two "
        "sources of randomness make a single measurement a random variable: "
        "which tokens are drawn from the text, and which subsamples are drawn "
        "within."))

    s.append(H("1.1 Two measures of estimator quality", 2))
    s.append(P(
        "Measurement noise is separated from genuine text-to-text differences "
        "by a one-way random-effects decomposition over N texts and k seeds, "
        "d<sub>ij</sub> = μ + a<sub>i</sub> + e<sub>ij</sub>. The within-text "
        "variance is estimated inside each text, where a<sub>i</sub> is constant "
        "and cancels, and averaged across texts; the variance of the per-text "
        "means is inflated by σ<sub>w</sub><super>2</super>/k, the standard "
        "error of the mean, and is debiased by subtracting it. Noise is reported relative to d, since d spans a "
        "factor of five across q."))
    s.append(P(
        "The second measure is the residual of the log-log fit, which lives in "
        "log space and is therefore already relative. The two are independent, "
        "and an estimate good by one can be bad by the other. Because "
        "d = α/(1 &#8722; b), a shift in the slope propagates to d amplified by a "
        "factor of d itself: at d = 21 a slope error of 0.005 gives a 10% error "
        "in d, where at d = 3.5 the same slope error gives 1.7%. Consequently "
        "the larger the measured dimension, the less it can be trusted, and a "
        "good fit does not compensate."))
    s += fig("variance_decomp.png",
             "<b>Figure 1.</b> Measurement quality against q, human texts, "
             "L = 256, under the original configuration. Left: ratio of "
             "text-to-text differences to measurement noise. Centre: the two "
             "components separately. Right: deviation from the power law. The "
             "two criteria disagree by construction — at high q_small the power "
             "law holds best of anywhere in the table (0.3%) while the noise is "
             "at its maximum, because there d is large and slope error is "
             "amplified accordingly; at high q_large the model itself breaks "
             "down (4.3%).")

    # ---------------------------------------------------------------- 2
    s.append(H("2. Estimator calibration"))

    s.append(H("2.1 A quantisation artefact in the published rule", 2))
    s.append(P(
        "The reference implementation discards k = floor(q·m) edges, where "
        "m = n - 1 varies across the fit grid. The realised trimmed fraction "
        "k/m therefore differs between subsample sizes: at the smallest "
        "subsample floor rounds q down by up to 1/m, less is discarded, S is "
        "inflated there, and the fitted slope tilts. The result is a saw-tooth "
        "in d(q) with troughs at q = 0.05, 0.15, 0.25, 0.35 and 0.45, of "
        "amplitude 0.5–0.9 in d. It is invisible on the original grid of step "
        "0.1, which lands only where floor is exact for every m, and it "
        "reproduces on uniform 8-dimensional synthetic data whose true curve "
        "must be smooth."))
    s.append(P(
        "Two remedies were compared against giving the boundary edge a "
        "fractional weight 1 - frac(q·m). Rounding instead of flooring removes "
        "only a floating-point special case: on a non-aligned grid it gives a "
        "roughness of 2.235 against 2.497 for flooring, where fractional "
        "weighting gives 0.127, because the artefact arises from k/m ≠ q rather "
        "than from the sign of the error. Aligning the grid so that every m is "
        "a multiple of 20 makes q·m an integer for the whole grid and removes "
        "the need to round at all; on such a grid rounding and fractional "
        "weighting agree exactly, which also establishes that fractional "
        "weighting is the correct continuation of exact integer trimming rather "
        "than a heuristic. Both are adopted: the grid is aligned, and "
        "fractional weighting is retained as a guard for q off the grid and for "
        "binary arithmetic, where for instance 0.35 · 180 evaluates to "
        "62.99999999999999 and floors to 62 instead of 63."))

    s.append(H("2.2 Configuration", 2))
    s.append(P(
        "The remaining hyperparameters were selected by a paired sweep on "
        "common texts. Relative noise of a single measurement falls from 0.114 "
        "under the published settings to 0.026."))
    s.append(table([
        ["Parameter", "Value", "Evidence"],
        ["Subsampling", "without replacement",
         "bootstrap leaves ~37% duplicate points at n = L, biasing d up 12–18%"],
        ["Pool", "2L tokens",
         "halves noise, leaves d unchanged (7.88 vs 7.89)"],
        ["Replicates", "32", "0.114 → 0.076; 64 gives 0.066 at twice the cost"],
        ["n grid", "aligned, 41…201",
         "grid shape is immaterial; grid range is not"],
        ["Trimming", "fractional", "see 2.1"],
        ["L", "201, absolute", "a fractional L reimports length dependence"],
        ["L against q", "independent", "rescaling costs the q axis its meaning"],
    ], widths=[3.1 * cm, 3.0 * cm, 10.3 * cm]))
    s.append(Spacer(1, 9))

    s.append(H("2.3 Dependence on the number of sampled tokens", 2))
    s.append(P(
        "The estimate follows d &#8733; L<super>γ</super> with γ ≈ &#8722;0.21 at "
        "q = 0 and up to &#8722;0.43 at q_small = 0.3. The reconfigured estimator barely changes the "
        "exponent (&#8722;0.24 and &#8722;0.49 before), which identifies this as structural "
        "rather than an artefact, and no convergence is reached within the "
        "accessible range: each doubling of L removes a further 8–10%. The "
        "interpretation is that d is estimated over the window n in [0.2L, L], "
        "and different L are different ranges of scale; since qPHD exists "
        "precisely to measure how dimension varies with scale, independence of "
        "the measurement window should not be expected. L must therefore be "
        "matched exactly, and absolute values of d carry no meaning without it. "
        "One near-invariant point exists, q_large ≈ 0.6, with γ = &#8722;0.13 and "
        "d(768)/d(256) = 1.01."))
    s += fig("length_dependence.png",
             "<b>Figure 2.</b> d against L on identical texts, log-log, under "
             "the published settings (left) and the reconfigured estimator "
             "(right). The slopes are nearly unchanged, and no plateau appears. "
             "q_large = 0.9 runs opposite in sign to everything else, so the "
             "direction of the dependence itself turns somewhere between "
             "q_large 0.6 and 0.9.")

    s.append(P(
        "With L fixed as an absolute count, the original length of the text no "
        "longer influences the result: regressing log d on log(token count) "
        "within genre over 160 texts of 322–881 tokens gives slopes between "
        "&#8722;0.014 and +0.031, indistinguishable from zero. Setting L as a "
        "fraction of the text instead reimports the dependence in full, with "
        "slopes of &#8722;0.13 to &#8722;0.56, as it must: if d &#8733; L<super>&#8722;0.2</super> and L is "
        "proportional to length, then d becomes a function of length."))
    s.append(P(
        "Scaling L with q so that the retained edge count is constant — the "
        "obvious remedy for the growing noise at high q — was tested and "
        "rejected. It achieves its aim, noise ceasing to grow and instead "
        "falling from 0.070 to 0.037, but the problem it addresses has already "
        "been removed: at fixed L = 228 the reconfigured estimator stays below "
        "0.068 across the whole range, against 0.228 originally. Meanwhile the "
        "rescaling bends the d(q) curve through the known L-dependence, and "
        "correcting the scaled arm back to a common L by L<super>0.21</super> "
        "reproduces the "
        "fixed-L arm (16.03 against 16.10 at q = 0.6), confirming that the "
        "difference carries no additional information."))

    # ---------------------------------------------------------------- 3
    s.append(H("3. Genres and generators"))
    s.append(P(
        "The calibrated estimator was applied to four genres and five sources, "
        "150 texts each, 3000 texts in total. Genre ordering among human texts "
        "is stable at low q — xsum 9.07, amazon_reviews 8.40, writingprompts "
        "8.14, academic_abstracts 7.46 — but the curves cross near q = 0.75: "
        "academic_abstracts starts lowest and reaches 19.15 by q = 0.9, above "
        "all others, while writingprompts saturates near 15 and then declines. "
        "A single value of q is therefore insufficient to characterise a genre; "
        "the shape of the curve carries information that no single point does."))
    s += fig("qphd_L201_sources_q_small.png",
             "<b>Figure 3.</b> qPHD against q by source within each genre, "
             "q_small. Human text (black) lies below every generator in every "
             "genre, at every q and in all three trimming modes, without "
             "exception. Bands are standard errors of the mean, which at "
             "n = 150 are about twelve times narrower than the text-to-text "
             "spread; they show that the group averages are well separated, not "
             "that individual texts are.")
    s.append(P(
        "Separability of individual texts is measured by the ROC-AUC of d "
        "between human and each model. Averaged over genres and models in the "
        "q0.5_range mode it rises from 0.775 at q = 0 to 0.831 at q = 0.35 "
        "before falling to 0.792 at q = 0.5, so the trimmed estimator is more "
        "informative than the untrimmed one, which corresponds to q = 0. Effect "
        "sizes behave the same way: Cohen's d for qwen3.6-flash rises from 2.23 "
        "at q = 0 to 2.44 at q = 0.3 and halves again by q = 0.9. The largest "
        "value observed is 0.994 for qwen3.6-flash in xsum."))
    s.append(P(
        "AUC here is a rank statistic of two samples rather than the accuracy "
        "of a fitted classifier, so no held-out data is required for it to be "
        "unbiased. The <i>choice</i> of the best q is made on the same data, "
        "and was therefore checked by split-half selection over 20 repeats: the "
        "selection bias averages +0.007 AUC and reaches +0.032 at most, the "
        "curve being smooth in q so that the effective number of independent "
        "attempts is far below the 57 grid points. Headline values are "
        "unchanged, and the third decimal is accordingly not reported."))
    s.append(P(
        "qwen3.6-flash separates about twice as strongly as the others, while "
        "deepseek-v4-flash, gpt-5-nano and gemini-2.5-flash form a tight group "
        "that these curves barely distinguish. The measure captures machine "
        "authorship well and the identity of the model poorly. Note that "
        "generator texts open with a human prefix of roughly sixty words that "
        "the model was asked to continue; it was retained, as the dataset is "
        "built that way, and works against the reported separation."))

    # ---------------------------------------------------------------- 4
    s.append(H("4. Which properties causally determine dimension"))
    s.append(P(
        "Thirty-one perturbations were applied to 50 texts per genre and "
        "compared against an unperturbed arm on the same text, so that "
        "between-text variance — the larger of the two variance components — "
        "drops out. Ten perturbations are mechanical and involve no model; "
        "twenty-one are LLM rewrites, 4000 calls in total, each holding length "
        "within about 10% and changing one named property."))

    s.append(H("4.1 Fidelity checking is not optional", 2))
    s.append(P(
        "Every rewrite was verified against a metric tracking its target "
        "property before any dimension was computed, the criterion being the "
        "share of texts moved in the intended direction, for which 0.5 is "
        "chance rather than half success. Two rewrites initially failed this "
        "check, and the consequence was not a null result but a result of the "
        "wrong sign. Instructions phrased in relative terms — 'more' or 'fewer' "
        "ideas — were satisfied by rewriting in general: both arms then raised "
        "d, because every LLM rewrite raises d on its own, as Section 3 shows "
        "independently. Restating the target as an absolute property checkable "
        "sentence by sentence, and moving to a stronger model, raised execution "
        "from 0.49 to 0.80 and reversed the measured effect from +19.2% to "
        "-40.0%. The topic pair failed and was repaired identically. Two cases "
        "out of two."))
    s.append(P(
        "Two metric choices had to be discarded along the way. A word-count "
        "measure of idea density rises when a text is shortened, by Heaps' law, "
        "and again when it is reworded, so it reported the opposite of the "
        "truth; it was replaced by the mean pairwise distance between sentence "
        "embeddings. Those raw cosines sit near 0.93 because the embedding "
        "space is anisotropic, leaving almost no dynamic range, so the vectors "
        "must be centred first — without centring every manipulation looks like "
        "a no-op."))

    s.append(H("4.2 Results", 2))
    s.append(P(
        "Evidence is graded by three criteria. A <i>reversal</i>, the two arms "
        "moving d in opposite directions, cannot be produced by a component "
        "common to both arms and so is evidence that the named property is "
        "responsible. <i>Replication</i> across two independent rewriting "
        "models guards against a quirk of one model's prose. A <i>mechanical "
        "control</i>, a direct manipulation with no model involved, is the "
        "strongest form available. Effect size is Cohen's dz of the paired "
        "differences; under normality dz maps onto the share of texts moving "
        "one way, and the mapping holds closely in these data (dz = 2.43 "
        "predicts 99.3% and 99.5% is observed)."))
    s += fig("causal_ranking.png",
             "<b>Figure 4.</b> Properties ranked by |dz|, coloured by the kind "
             "of evidence available. Dotted lines mark the share of texts "
             "moving in one direction that each dz corresponds to.", width=12.6)

    s += fig("perturb_contrasts.png",
             "<b>Figure 5.</b> Each up-arm minus its down-arm, which cancels "
             "the component shared by all rewrites. In q_large both leading "
             "factors cross zero near q = 0.8 and turn negative: at the finest "
             "scale a more varied text measures as less dimensional. That is "
             "also the region where the power law itself breaks down, so it is "
             "recorded as an observation to be checked rather than a result.")
    s += fig("perturb_mechanical.png",
             "<b>Figure 6.</b> The mechanical perturbations alone. Three tiers "
             "are visible: shuffling and punctuation stripping at 10–44%, typos "
             "and line breaks at 2–9%, and sentence-length and sentence-order "
             "changes below 3% everywhere. Shuffling words across the text and "
             "shuffling them within sentences coincide almost exactly, while "
             "reordering whole sentences with word order intact is flat, which "
             "places the effect on word order inside a sentence rather than on "
             "the coherence of the text.")

    s.append(P(
        "Four factors emerge with strong support. <b>Lexical diversity</b> is "
        "the largest, dz = 2.43, peaking in q0.5_range at q = 0.25, reversing "
        "on both models and asymmetric: raising diversity moves d three times "
        "further than lowering it. Substituting rare words gives +25.7% where "
        "substituting common ones gives &#8722;7.2%, so the operative property is "
        "non-repetition rather than rarity. <b>Word order within a sentence</b> "
        "follows at dz = 2.17 under mechanical control, peaking in q_large at "
        "q = 0.7 and changing sign across the scale: &#8722;11.2% at q_small = 0.9 "
        "against +44.1% at q_large = 0.9. <b>Topic variety</b> reaches "
        "dz = 1.93 and <b>idea count</b> dz = 1.82, the latter being the only "
        "factor producing a large decrease, &#8722;47.5%, with an asymmetry opposite "
        "to the lexical one: human text sits near a ceiling in idea density but "
        "has room to fall."))
    s.append(P(
        "Sentence length is not among them. It was the natural hypothesis, "
        "since texts at the low end of the dimension distribution proved to be "
        "dialogue and verse with short sentences, correlating with sentence "
        "count at &#8722;0.43. Manipulated directly, however, it moves d by about 1%, "
        "so the correlation was not causal; the ±14% seen through an LLM "
        "instructed to simplify syntax runs through the vocabulary it changes "
        "along the way. Typography behaves likewise, line breaks giving &#8722;3.8% "
        "and capitalisation &#8722;0.7%, though the latter had correlated at &#8722;0.51."))
    s.append(P(
        "Peak locations divide the q axis cleanly. All five semantic properties "
        "peak in q0.5_range between q = 0.20 and 0.35; all three structural ones "
        "peak in q_large between 0.7 and 0.8, with typos in between. The axis "
        "appears to separate what is written from how it is arranged."))

    s.append(H("4.3 Do the two structural manipulations share a channel?", 2))
    s.append(P(
        "Punctuation stripping and word shuffling peak in the same regime at "
        "nearly the same q, and their effect profiles correlate at 0.61, "
        "suggesting a single channel: the dismantling of local organisation. "
        "The suggestion is testable, since one shared resource implies that "
        "applying both lands near the larger of the two rather than near their "
        "sum. Two composite perturbations were run."))
    s.append(P(
        "The hypothesis is rejected in the working range. At q = 0.7 the pair "
        "gives +67.2% where the larger alone gives +34.2%. Under a "
        "multiplicative null — effects on a ratio quantity composing as "
        "products — the two are independent to within 2 percentage points "
        "across q = 0.5–0.7. That last statement depends on the null: a "
        "slope-additive null is exact at q = 0 and overshoots by 33 points at "
        "q = 0.7, and nothing in the geometry dictates which to use. Neither the "
        "rejection nor the saturation that appears by q = 0.9, where both nulls "
        "overpredict by 27 and 57 points, depends on that choice. The "
        "saturation is independent evidence that this regime measures local "
        "organisation specifically: once the first manipulation has run, the "
        "second has little left to destroy."))
    s.append(P(
        "The mechanism behind punctuation stripping is known: the deleted "
        "tokens sit 1.5 times closer to their nearest neighbour than the rest, "
        "11.5 against 17.4, and so carry the shortest MST edges that q_large "
        "sums over. This explains the effect rather than diminishing it. What "
        "the experiment does not establish is the wider claim that punctuation "
        "as a feature of style governs dimension, since the manipulation also "
        "fragments words; testing that would require substituting or inserting "
        "marks rather than deleting tokens."))

    # ---------------------------------------------------------------- 5
    s.append(H("5. Limitations"))
    s.append(P(
        "All results rest on one embedding model, ModernBERT-base, and the "
        "semantic fidelity metric is built on the same vectors as qPHD itself, "
        "so that check is not fully independent of what it verifies. The two "
        "rewriting models differ in force as well as in reliability — gemini "
        "moved type-token ratio by +40.5% where sonnet moved it +14.2%, both at "
        "full execution — so effect magnitudes are comparable only within a "
        "model. The sign reversal at q_large ≈ 0.8 falls in the region where "
        "the power law itself degrades, with residuals of 4.3% against 0.3% in "
        "the working range, and requires separate verification. Finally, "
        "d &#8733; L<super>&#8722;0.21</super> does not converge within the reachable range of L, so "
        "absolute dimensions are properties of a measurement window and not of "
        "a text."))

    s.append(H("6. Reproduction"))
    s.append(P(
        "Code, per-text results, generated texts and figures are committed. "
        "The calibrated configuration is fixed in <font face='Courier' "
        "size='8.6'>scripts/config.py</font> with the measurement justifying "
        "each choice; the reference implementation is retained unmodified in "
        "<font face='Courier' size='8.6'>scripts/reference/</font> for "
        "line-by-line comparison, and the published settings remain reachable "
        "through a flag. The parameter sweeps that selected the configuration "
        "are separated into <font face='Courier' size='8.6'>experiments/</font>, "
        "indexed by the question each answers."))
    return s


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=2.3 * cm, rightMargin=2.3 * cm,
        topMargin=2.0 * cm, bottomMargin=1.9 * cm,
        title="Multi-scale intrinsic dimension of text",
        author="qPHD extension",
    )
    doc.build(story(), onFirstPage=page_number, onLaterPages=page_number)
    print("saved", OUT)


if __name__ == "__main__":
    main()
