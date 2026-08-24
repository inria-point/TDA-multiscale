#!/bin/bash
# Stage 2 end to end. Safe to re-run: OpenRouter responses are cached per call,
# so a repeat costs nothing for what already succeeded and only fills the gaps.
cd "$(dirname "$0")"
P=.venv/bin/python

echo "[1/3] ждём текущую генерацию стилей"
while pgrep -f "llm_edits.py --corpus coling" > /dev/null; do sleep 20; done

echo "[2/3] добираем пропущенное (кэш делает повтор бесплатным)"
$P scripts/llm_edits.py --corpus coling --n-texts 120 --min-words 280 \
  --model anthropic/claude-sonnet-4.5 --workers 8 \
  --out results/coling_style_edits.json \
  --perturbations style_news_telegraphic style_simplified style_news_dates_suffixes \
  style_dialogue_interjections style_literary style_bulletin_abbreviations \
  style_scientific_terms >> results/coling_style_edits.log 2>&1

echo "[3/3] qPHD по всем пертурбациям (интернет не нужен)"
$P scripts/run_perturbations.py --corpus coling --n-texts 120 \
  --texts-json results/coling_style_edits.json --tag _coling \
  --perturbations identity \
    burst_alternate burst_flatten drop_sentence_boundaries loop_phrase \
    add_punctuation strip_punctuation add_digits strip_digits \
    capitalize_terms lowercase to_numbered_list drop_function_words \
    shuffle_words shuffle_within_sentences add_typos \
    style_news_telegraphic style_simplified style_news_dates_suffixes \
    style_dialogue_interjections style_literary style_bulletin_abbreviations \
    style_scientific_terms \
  > results/run_perturb_coling.log 2>&1

echo "готово: results/perturb_L201_coling.csv.gz"
