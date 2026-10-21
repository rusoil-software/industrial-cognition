# Detector benchmark: is OWLv2 the wrong tool?

The question this document answers is not "which model is best" in the abstract. It is
narrower and more awkward: **on our own station and our own warehouse photograph, what does
it cost to keep OWLv2, and what would replacing it buy?** The answer has to survive a licence
review, a CPU-only deployment and a fixed camera, so the numbers below are measured through the
same gate the pipeline ships, on the same encoded frames the pipeline sends, with the same
software stack the API uses.

Nothing in this document is an opinion about model quality. Every row in
`reports/detector_benchmark.csv` comes from `scripts/benchmark_detectors.py`, and the command
that produced it is printed at the end.

## The protocol

Thirteen samples, four labels, in one CSV:

| scenario | samples | what it asks |
| --- | --- | --- |
| station, part present | 8 | one pan phase per eighth of the replay period, `steel blank` |
| station, part absent | 4 | wall band, floor band and the two racking margins of the same photograph |
| warehouse | 1 | `robot`, `person`, `cardboard box` in one scene |

Three choices make the comparison fair rather than flattering:

1. **The frames are the deployment's frames.** Every image goes through
   `encode_jpeg(..., max_side=960, quality=80)` and back before any model sees it
   (`src/benchmark/dataset.py`, `deployment_frame`). Full-resolution photographs add 2-4
   points of confidence to *every* model, and the deployment cannot produce them.
2. **The station phrase is the one we ship.** `steel blank`, not `metal sheet`. Measured
   both ways on the incumbent: `metal sheet` gives recall `0.25` and a mean sheet IoU of
   `0.43`; `steel blank` gives recall `1.00` and `0.86`. A benchmark that measures the
   easier phrase measures nothing.
3. **The presence answer goes through the shipped gate.** `score_sample` calls the real
   `PresenceGate` from `src/vision/presence.py`, so a model that puts a confident box on the
   table fails the same way it would in production.

Latency is the wall-clock time of `detect(image, labels)` for the whole sample, including any
per-label work the adapter has to do. `fps` is `1 / mean latency`, so a model that needs three
forward passes for three labels pays for them. Load time and peak RSS/VRAM are measured
separately, because a 20 s load and 2.8 GB of resident memory are costs a small node notices.

## Ground truth

Hand-labelled with a 0.1 grid overlay, deliberately generous - they describe the object, not
one model's output:

| sample | label | box (x, y, w, h) |
| --- | --- | --- |
| `metal_sheet_test_image.jpg` | `steel blank` | 0.335, 0.455, 0.320, 0.220 |
| `roboflow_test_detect_human_robot.webp` | `robot` | 0.280, 0.170, 0.470, 0.250 |
| `roboflow_test_detect_human_robot.webp` | `person` | 0.520, 0.510, 0.140, 0.130 |
| `roboflow_test_detect_human_robot.webp` | `cardboard box` | 0.655, 0.335, 0.135, 0.070 |

A detection counts as a hit at IoU >= 0.30 (`HIT_IOU`). The presence answer is scored
separately: recall over the eight positives, specificity over the four absence crops.

## Results

Measured 2026-10-21 on the development rig (RTX 4060 Laptop, 8 GiB) inside
`industrial-cognition/benchmark:dev`, 13 samples, `reports/detector_benchmark.csv`. "Present" is
the presence question from section 19, scored through the gate the cluster ships.

| model | licence | weights | p50 | fps | peak RSS / VRAM | present | sheet IoU | warehouse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OWLv2 (ONNX, CPU) | Apache-2.0 | 293 MB | 5 944 ms | 0.14 | 2.8 GB / - | **100%** | **0.86** | hit |
| YOLO-World v2 M (GPU) | AGPL-3.0 | 55 MB | **20 ms** | **32.5** | 3.5 GB / 600 MB | 0% | 0.00 | hit |
| YOLO-World v2 L (GPU) | AGPL-3.0 | 90 MB | 23 ms | 46.0 | 3.7 GB / 919 MB | 0% | 0.00 | hit |

Read the table the way the task is phrased, not the way a leaderboard reads it:

* **The fast models answer the wrong question.** Both YOLO-World variants see the warehouse
  (22 and 45 boxes, best IoU 0.73) and return *no box at all* for the metal blank - not a wrong
  box, not a low score: nothing, at confidence 0.002 and above. A bare reflective sheet is
  simply not in its vocabulary of nouns, and a 300x faster model that cannot answer the question
  is not 300x better.
* **The slow model answers it with confidence ~0.10.** OWLv2 finds the sheet in all eight pan
  phases (IoU 0.84-0.92) and refuses all four absence crops, but its chosen confidence is 0.053-0.075
  against a table box at 0.13-0.16. That thin margin is why the presence gate exists and why it
  is geometry that separates the part from the table.
* **Speed alone does not close the gap, because the gap is not speed.** Dropping to 0.14 fps on
  CPU is a sampling problem the pipeline already solves (`PIPELINE_DISPATCH_INTERVAL_SECONDS`);
  being blind to the workpiece is not solvable by sampling more often.

The honest conclusion for a production decision: **stay on OWLv2 for now**, and treat "a
permissively licensed open-vocabulary detector that is both fast and able to see a bare metal
blank" as an open question - not as a settled comparison, because the two models that could
answer it (Grounding DINO base, Florence-2-base) could not be downloaded here.

A vocabulary change is a configuration cost, not a frame cost: the YOLO-World adapter rebuilds
its CLIP text embeddings on the first frame after `set_classes()`, measured at **6-9 s**. The
benchmark groups samples by vocabulary and warms each group untimed, so the CSV reports per-frame
detection latency; a deployment that rewrote its prompt every frame would pay that cost every
frame, which is why prompts are per camera and changed by an operator, not per frame.

## Licences, which decide this as much as the numbers

| model | weights licence | usable in a shipped product? |
| --- | --- | --- |
| OWLv2 (ONNX export) | Apache-2.0 | yes |
| Grounding DINO base (Swin-T) | Apache-2.0 | yes |
| Florence-2-base | MIT | yes |
| YOLO-World M / L | AGPL-3.0 | **no** - copyleft reaches the served service |
| Grounding DINO 1.5 Edge | proprietary | no open weights; API only |
| DINO-X | proprietary | no open weights; API only |

Two consequences that the licence column forces:

* **The fastest model on the list cannot be used.** YOLO-World is the only candidate that is
  genuinely real-time on this hardware, and AGPL-3.0 is not something this project can ship
  behind an HTTP API without opening the whole service. It stays a reference point, not an
  option.
* **The remaining candidates are the slow, heavy zero-shot models** - which is the same
  family OWLv2 belongs to. A replacement therefore has to win on *accuracy at the station*,
  not on speed, because no permissively licensed model here is fast enough to remove the
  sampling in `PIPELINE_DISPATCH_INTERVAL_SECONDS`.

## Models that could not be measured, and why

Honesty is part of the deliverable: a row that says "not measured" is worth more than a row
invented from a blog post.

| model | status | reason |
| --- | --- | --- |
| Grounding DINO 1.5 Edge | not measurable | no open weights; IDEA Research serves it behind an API and credentials |
| DINO-X | not measurable | closed API, same |
| Grounding DINO base | **blocked by the network** | HuggingFace CDN measured at ~30 KB/s from this host (694 MB of weights, timed out repeatedly); the adapter is written and its row appears as `measured=no` |
| Florence-2-base | **blocked by the network** | same, 464 MB |

The network limit is real and reproducible, not a model problem: a 57 MB GitHub release
downloads at 4 MB/s on the same machine, while `huggingface.co` ranges stall at 30 KB/s with
`The read operation timed out`. Re-running the command when the network allows it is the whole
fix - there is no code change behind these two rows.

## Running it

```bash
# once, on a machine with a GPU (and a Docker daemon that can pass it through)
docker build -f src/benchmark/Dockerfile -t industrial-cognition/benchmark:dev .

# the measurement: repo mounted, weights cached outside the checkout
docker run --rm --gpus all \
    -v "$PWD:/work" -w /work \
    -v "$HOME/.cache/ic-benchmark:/cache" \
    -e HF_HOME=/cache/hf -e YOLO_WEIGHTS_DIR=/cache/yolo \
    industrial-cognition/benchmark:dev \
    python scripts/benchmark_detectors.py --models all \
        --out reports/detector_benchmark.csv
```

The script writes three things: the CSV (one row per model, the evidence), a details CSV
(one row per sample, so a surprising average can be traced to the frame that caused it) and a
fixed-width table on stdout. `--device cpu` forces the CPU path, `--json` writes the rows for a
later comparison, `--models owl2-onnx,yolo-world-m` narrows the run.

Two operational notes:

* **The image is 11 GB and is not part of any production image.** `requirements/benchmark.txt`
  is installed nowhere else; the AGPL runtime and PyTorch stay out of the API and the workers.
* **Weights are cached outside the repository** (`YOLO_WEIGHTS_DIR`, `HF_HOME`), because the
  ultralytics default writes a 55 MB `.pt` into the working directory - which, inside the
  container, is the mounted checkout. Its CLIP text encoder (354 MB, `weights/ViT-B-32.pt`) is
  written there too and is git-ignored for the same reason.

## What this benchmark does not measure

* **Training.** These are zero-shot models used as shipped; fine-tuning OWLv2 or Florence-2 on
  a few hundred photographs of the stand is a separate, larger project.
* **Detection of moving parts.** The station samples are a still photograph panned 24 px, not
  video. Temporal behaviour is covered by the tracker (`doc/vision-pipeline.md` §18).
* **Cost at scale.** One node, one GPU, one process. A production question ("how many cameras
  per GPU") needs the numbers from §16 and a real camera count.
* **Quality of the boxes on other scenes.** Two photographs is a protocol, not a dataset.

Reference: `src/benchmark/` (the harness), `doc/vision-pipeline.md` §19 (the station gate this
benchmark scores through), `doc/camera.md` (the station camera and its prompts).
