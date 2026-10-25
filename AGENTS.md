# Agent memory for `industrial-cognition`

Project-specific notes that a new session should read before touching the detector work.
Written 2026-10-25 from the session that built `notebooks/Florence_DINO_OWL_bench.ipynb`
and `notebooks/REPORT.md`.

## Pinned environment (do not "upgrade to fix it")

| Fact | Value |
| --- | --- |
| Python | 3.11.9, interpreter at `.venv/` |
| `transformers` | **4.49.0** — the version everything here is written against |
| torch | 2.14.x+cu132, RTX 4060 Laptop, 8 GB |
| Pins live in | `requirements/benchmark.txt` |

Newer `transformers` breaks this work in both directions: 5.x rejects the Florence-2 remote
code and the `OmdetTurbo`/`OwlV2` call shapes used here, and the Grounding-DINO fp16 bug below
is version-specific. Install nothing; adapt the code to the pinned stack instead.

## Where the detector evidence lives

- `doc/detector-benchmark.md` — the **production** benchmark: 13 deployment frames through the
  shipped gate, OWLv2 vs YOLO-World, plus the licence table. Conclusion: stay on OWLv2.
  Grounding DINO base and Florence-2-base are listed there as *not measured* (HF CDN stall).
- `notebooks/Florence_DINO_OWL_bench.ipynb` — the **follow-up** that measures those two plus
  OWLv2 on a 20-image calibration set (`assets/images/calibration/{positive,negative}`), with
  13 scored arms, an AUC/Youden report and a prompt sweep.
- `notebooks/REPORT.md` — the write-up: models, defects found, results, verdicts, and the
  licence survey. `notebooks/scores_report.csv`, `prompt_sweep.csv`, `benchmark_results.csv`
  and the `detection_grid_*.png` files are its evidence.

Headline numbers (20 images, balanced, best operating point per arm): OWLv2 @1280 75 % acc /
MCC 0.577 / LOO 70 %; OWLv2 @960 70 % / 0.500 (best ranking, AUC 0.740); Grounding DINO
60 % / 0.250; Florence-2 caption-presence 50 % / 0.000.

## The findings that change decisions

1. **The ceiling is the dataset, not the model.** Positives are a bright sheet on a scuffed
   plate; `negative/09.png` and `negative/10.png` contain *paper* sheets of the same shape.
   The class difference is **material**, which no text prompt expresses. No model swap and no
   threshold fixes that.
2. **These models answer "where", not "whether".** All three are localisers. Presence has to be
   manufactured from a score plus a threshold, and the threshold must be re-derived per model,
   per camera. Report MCC/TNR and a leave-one-out number, never accuracy alone: on a balanced
   set 50 % is what "always yes" scores.
3. **`steel blank` is untested here and probably better than `steel sheet`.** The station
   benchmark measured `metal sheet` → recall 0.25 / IoU 0.43 versus `steel blank` → 1.00 / 0.86.
   Cheapest remaining experiment.
4. **Verdict:** case A (data as-is) → OWLv2-base at native 1280x1024, prompt `steel sheet`,
   τ ≈ 0.09. Case B (recaptured so the sheet has a signature nothing else shares) → Grounding
   DINO tiny. Florence-2 only through `<MORE_DETAILED_CAPTION>` presence, and only if captions
   are accurate on your frames — on this data they say "a rectangular piece of paper".
5. **Fine-tuning is not the shortcut.** It needs boxes that do not exist (20 image-level labels
   only), so annotation dominates the cost. Fix the capture first. See `REPORT.md` §11.

## `transformers` 4.49 quirks that cost hours

- **Grounding DINO is not fp16-safe.** `modeling_grounding_dino.py` `get_text_position_embeddings`
  casts the text position embedding to `.float()`; adding it to half features raises
  `mat1 and mat2 must have the same dtype, but got Float and Half`. Load it with
  `torch_dtype=torch.float32`.
- **OWLv2 cannot change input resolution without a flag.** The position-embedding table is frozen
  at 3601 rows (60x60 patches for 960/16). A 1280x1024 frame yields 5121 tokens and crashes
  `size of tensor a (5121) must match … b (3601)`. Pass `interpolate_pos_encoding=True` to the
  model call; it is a no-op at 960.
- **Florence-2 `<OPEN_VOCABULARY_DETECTION>` returns `bboxes_labels`, not `labels`.** The
  `labels` key only exists for `<OD>`-style tasks. Reading the wrong key silently yields empty
  labels and a presence filter that can never fire.
- **`<OPEN_VOCABULARY_DETECTION>` cannot answer "absent".** Its prompt template is "Locate
  {input} in the image." — an instruction that presupposes the object. It returned a box on
  20/20 negatives and its score is the LM probability of the written tokens.
- **Florence-2 has no VQA head.** The released checkpoints exclude it even though the paper
  reports it. A question-style prompt cannot produce "no".
- **`generate()` returns a bare tensor** unless `return_dict_in_generate=True`; code that reads
  `out.sequences` must pass it.
- Florence-2 `generate` consumes `inputs_embeds`, so `out.sequences` starts at BOS — the prompt
  is not echoed. Do not compute a decoder prefix length from `input_ids`.

## Working on the notebook

- It is a **tracked artifact with saved outputs**; those outputs are evidence quoted by
  `REPORT.md`. Do not clear them casually.
- **A running JupyterLab kernel will overwrite your edits.** It holds the notebook in memory and
  saves it wholesale. Before editing, ask the user to close the tab or do File → Reload Notebook
  from Disk; after editing, re-read or `grep` to confirm the change survived.
- Edits are JSON string edits: each source line is `        "code\n",` with 8-space indent, the
  last line of a source array has no trailing comma, and blank code lines are `"\n",`.
  Anchor replacements on a unique neighbouring line — `out = m(**inputs)` alone occurs 3 times.
- Do not run the notebook from an agent session; the GPU is shared with the user's kernel.

## Repo conventions

- `doc/conventions.md` mandates documentation-first: doc in `doc/`, then code.
- Reports belong with their evidence (`notebooks/REPORT.md` next to the CSVs and grids).
- Do not commit: `.venv/**`, `weights/`, `reports/*.json`, `sample_data/` (see `.gitignore`).
