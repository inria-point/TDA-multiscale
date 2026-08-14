# Обзор: графики и результаты

Одностраничная навигация по всему, что посчитано. Подробности — в
[README.md](README.md), связный текст — в [report/qphd_report.pdf](report/qphd_report.pdf).

Актуальная конфигурация оценки: L = 201, выровненная сетка n, дробная обрезка, подвыборки
без возвращения из пула 2L, 32 повтора ([scripts/config.py](scripts/config.py)).

---

## Графики

| график | что на нём | главное |
|---|---|---|
| [variance_decomp.png](figures/variance_decomp.png) | шум оценки, различия между текстами и невязка по q | два критерия качества расходятся: при q_small = 0.9 фит лучший, а шум максимальный |
| [length_dependence.png](figures/length_dependence.png) | d против L, одни и те же 16 текстов, лог-лог | d ∝ L^−0.21, не сходится; настройка оценки почти не меняет показатель |
| [length_policy.png](figures/length_policy.png) | d против длины текста при 4 политиках выбора L | абсолютный L даёт наклон ≈ 0, долевой возвращает зависимость (−0.13…−0.56) |
| [L_per_q.png](figures/L_per_q.png) | масштабирование L как N/(1−q) против фиксированного | шум выравнивает, но ось q теряет смысл; расхождение объясняется зависимостью от L |
| [human_L201_v2_curves.png](figures/human_L201_v2_curves.png) | кривые qPHD(q) по 4 жанрам, human, 150 текстов | порядок жанров меняется около q ≈ 0.75: одной точки q недостаточно |
| [qphd_L201_sources_q_small.png](figures/qphd_L201_sources_q_small.png) | кривые по 5 источникам внутри каждого жанра | human ниже всех генераторов везде, без исключений |
| [qphd_L201_auc_q0.5_range.png](figures/qphd_L201_auc_q0.5_range.png) | ROC-AUC human против каждой модели по q | оптимум при q ≈ 0.3, не при q = 0; максимум ~0.99 |
| [causal_ranking.png](figures/causal_ranking.png) | свойства по \|dz\|, цвет — тип доказательства | четвёрка: лексика 2.43, порядок слов 2.17, темы 1.93, идеи 1.82 |
| [perturb_effects_q_small.png](figures/perturb_effects_q_small.png) | все модификации по группам, крупный масштаб | лексическое разнообразие +57.8% при q = 0.2 |
| [perturb_effects_q_large.png](figures/perturb_effects_q_large.png) | то же, мелкий масштаб | перемешивание слов +44.1%; знак противоположен крупному масштабу |
| [perturb_mechanical.png](figures/perturb_mechanical.png) | только механические модификации, три режима | три яруса силы; порядок слов внутри предложения решает, порядок предложений — нет |
| [perturb_contrasts.png](figures/perturb_contrasts.png) | разности плеч, без общего эффекта переписывания | при q_large ≈ 0.8 главные факторы меняют знак |

Устаревшие, оставлены для истории: `human_L256*.png` (до перехода на L = 201),
`perturb_v2_effects_*.png` (только пара «идеи», до склейки прогонов),
`qphd_L201_{auc,sources}_{q_large,q0.5_range}.png` — те же данные в других режимах.

---

## Эксперименты

### Калибровка оценки

| скрипт | вопрос | ответ | данные |
|---|---|---|---|
| [variant_sweep.py](experiments/variant_sweep.py) | какие настройки снижают шум | без возвращения + пул 2L + 32 повтора: 0.114 → 0.026 | [variant_sweep_L256.csv](results/variant_sweep_L256.csv) |
| [variance_decomp.py](experiments/variance_decomp.py) | сколько в разбросе шума, сколько сигнала | σ_w и σ_b раздельно; шум ≈ 10% при L = 256 | [variance_decomp_human_L*.csv](results/) |
| [exp_grid_alignment.py](experiments/exp_grid_alignment.py) | можно ли убрать округление выбором сетки | можно, но дробная обрезка нужна всё равно; `round` не спасает (2.235 против 2.497 и 0.127) | [exp_grid_alignment.csv](results/exp_grid_alignment.csv) |
| [length_sweep.py](experiments/length_sweep.py) | как d зависит от L | d ∝ L^−0.21, не лечится настройкой | [length_sweep.csv](results/length_sweep.csv) |
| [exp_length_policy.py](experiments/exp_length_policy.py) | абсолютный L или доля от длины | только абсолютный | [exp_length_policy_summary.csv](results/exp_length_policy_summary.csv) |
| [exp_L_per_q.py](experiments/exp_L_per_q.py) | менять ли L вместе с q | не менять | [exp_L_per_q_summary.csv](results/exp_L_per_q_summary.csv) |
| [inspect_extremes.py](experiments/inspect_extremes.py) | чем отличаются тексты с крайними d | структурной повторяемостью, не лексикой (гипотеза позже опровергнута прямой манипуляцией) | [extremes_*.csv](results/) |

### Жанры и генераторы

| прогон | объём | результат | данные |
|---|---|---|---|
| [run_human_baseline.py](scripts/run_human_baseline.py) | 4 жанра × 150 текстов | кривые по жанрам, пересечение при q ≈ 0.75 | [human_L201_v2.csv](results/human_L201_v2.csv) |
| [run_qphd.py](scripts/run_qphd.py) | 5 источников × 4 жанра × 150 = 3000 текстов | все генераторы выше human; оптимум различимости q ≈ 0.3 | [qphd_L201.csv.gz](results/qphd_L201.csv.gz), [qphd_L201_auc.csv](results/qphd_L201_auc.csv) |

### Модификации текста

| прогон | объём | результат | данные |
|---|---|---|---|
| [llm_edits.py](scripts/llm_edits.py) | 4000 запросов, gemini-3.1-flash-lite и claude-sonnet-4.5 | 21 LLM-модификация | [llm_edits*.json](results/) |
| [check_edits.py](scripts/check_edits.py) | проверка исполнения до счёта d | 8 из 10 на 0.90–1.00; `ideas_up` 0.49 — не исполнена | [edit_fidelity.csv](results/edit_fidelity.csv) |
| [run_perturbations.py](scripts/run_perturbations.py) | 31 модификация × 4 жанра × 50 текстов | парные эффекты | [perturb_L201_all.csv.gz](results/perturb_L201_all.csv.gz), [perturbed_texts.json](results/perturbed_texts.json) |
| [analyze_perturbations.py](scripts/analyze_perturbations.py) | разбор, включая противопоставления плеч | таблица эффектов и контрастов | [perturb_effects.csv](results/perturb_effects.csv), [perturb_contrasts.csv](results/perturb_contrasts.csv) |
| [causal_summary.py](scripts/causal_summary.py) | ранжирование по силе доказательства | четвёрка причинных факторов | [causal_summary.csv](results/causal_summary.csv) |
| тест общего канала | пунктуация + перемешивание вместе | каналы независимы до q ≈ 0.7, насыщение к 0.9 | [perturb_L201_combo.csv.gz](results/perturb_L201_combo.csv.gz) |

---

## Сводка выводов

| № | вывод | где подробно |
|---|---|---|
| 1 | `floor(q·m)` даёт пилообразный артефакт на сетке q мельче 0.1 | README, п. 1 |
| 2 | Порядок жанров зависит от масштаба, кривые пересекаются | README, п. 2 |
| 3 | Тексты с низкой d — диалоги и верлибр (корреляция, не причина) | README, п. 3 |
| 4 | Хороший фит не гарантирует надёжной d: ошибка наклона усиливается в d раз | README, п. 4 |
| 5 | d ∝ L^−0.21, не сходится, настройкой не устраняется | README, п. 5 |
| 6 | При абсолютном L длина текста на d не влияет | README, п. 6 |
| 7 | Масштабировать L по q не нужно | README, п. 7 |
| 8 | Все генераторы выше human; оптимум различимости q ≈ 0.3, а не 0 | README, п. 8 |
| 9 | Причинные факторы: лексика, порядок слов, темы, число идей | README, п. 9 |
| 10 | Слабо исполненная инструкция даёт результат с **неверным знаком**, а не нулевой | README, п. 9 |
| 11 | Любое машинное переписывание само поднимает d — сравнивать надо плечи между собой | README, п. 9 |
