# Test assets

| File | Source | Licence | Used by |
| --- | --- | --- | --- |
| `roboflow_test_detect_human_robot.webp` | Roboflow test dataset (human + robot detection) [robo-environment-n6d0b](https://universe.roboflow.com/tamilvanan-jj11e/robo-environment-n6d0b) | Apache 2.0 | `tests/vision/test_detection_quality.py` |
| `metal_sheet_test_image.jpg` | Photograph taken by the project owner, 2026-10-21 (Redmi Note 12S) | Project asset, internal use | `tests/vision/test_metal_sheet_presence.py`, presence calibration |

The warehouse image is an overhead scene chosen because its two subjects are
**geometrically separable**: a yellow articulated robot arm at the top centre and a
worker in a hi-vis vest below it. That makes it possible to assert something stronger
than "detection returned a box" - namely that the prompt changes *where* the model
looks.

The metal-sheet photograph is the opposite kind of asset: a single subject, shot the way
a station camera would see it, and therefore useful for the question the deployment
actually asks - *is the workpiece there or not*. It is deliberately unglamorous: a
brushed sheet on a white table in a tiled hallway, with the floor, wall and doorway
that a fixed camera cannot avoid. Those extra surfaces are what make the absence
measurement meaningful; see `doc/vision-pipeline.md` §19 for the measured scores and the
tiled-floor false positive that decides how such a camera must be aimed.

Prompt fixtures live in `../prompts/*.json`: real CLIP tokenisations produced by
`scripts/encode_text_queries.py`, committed so the vision image (which carries ONNX
Runtime but deliberately not `transformers`) can run the tests. `steel_blank.json` and
`metal_blank.json` were encoded for the metal-sheet work; `steel blank` is the industrial
term for a cut sheet and scores noticeably higher than `metal sheet` on the real
photograph.

For a broader evaluation later, <https://huggingface.co/datasets/POSE-Lab/IndustryShapes>
is the intended next dataset.
