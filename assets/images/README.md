# Test assets

| File | Source | Licence | Used by |
| --- | --- | --- | --- |
| `roboflow_test_detect_human_robot.webp` | Roboflow test dataset (human + robot detection) [robo-environment-n6d0b](https://universe.roboflow.com/tamilvanan-jj11e/robo-environment-n6d0b) | Apache 2.0 | `tests/vision/test_detection_quality.py` |

The image is an overhead warehouse scene chosen because its two subjects are
**geometrically separable**: a yellow articulated robot arm at the top centre and a
worker in a hi-vis vest below it. That makes it possible to assert something stronger
than "detection returned a box" - namely that the prompt changes *where* the model
looks.

Prompt fixtures live in `../prompts/*.json`: real CLIP tokenisations produced by
`scripts/encode_text_queries.py`, committed so the vision image (which carries ONNX
Runtime but deliberately not `transformers`) can run the test.

For a broader evaluation later, <https://huggingface.co/datasets/POSE-Lab/IndustryShapes>
is the intended next dataset.
