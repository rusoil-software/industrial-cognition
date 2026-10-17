"""Does the artifact accept a *batched* text input and return one query per label?

The pipeline assumed a single class because the graph reports logits (1, 3600, 1) for a
batch-of-one text input. scripts/proof_of_work.py tokenises the whole label list at once
and takes argmax over the query axis, which implies the class dimension grows with the
text batch. This checks the artifact directly.
"""

import numpy as np
import onnxruntime as ort

MODEL = r"src\owl\models\cuda\owl2_model.onnx"

ROBOT = [49406, 8797, 269, 49407] + [0] * 12
HUMAN = [49406, 2751, 269, 49407] + [0] * 12
BOX = [49406, 19845, 2063, 269, 49407] + [0] * 11
MASK4 = [1, 1, 1, 1] + [0] * 12
MASK5 = [1, 1, 1, 1, 1] + [0] * 11


def main() -> None:
    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    print("inputs :", [(i.name, i.shape) for i in session.get_inputs()])
    print("outputs:", [(o.name, o.shape) for o in session.get_outputs()])

    pixel_values = np.zeros((1, 3, 960, 960), dtype=np.float32)

    cases = {
        "1 query (robot)": ([ROBOT], [MASK4]),
        "2 queries (robot, human)": ([ROBOT, HUMAN], [MASK4, MASK4]),
        "3 queries (robot, human, box)": ([ROBOT, HUMAN, BOX], [MASK4, MASK4, MASK5]),
    }

    for name, (ids_rows, mask_rows) in cases.items():
        ids = np.asarray(ids_rows, dtype=np.int64)
        mask = np.asarray(mask_rows, dtype=np.int64)
        try:
            outputs = session.run(
                None,
                {"pixel_values": pixel_values, "input_ids": ids, "attention_mask": mask},
            )
        except Exception as exc:  # noqa: BLE001 - the point is to see the failure
            print(f"{name:<32} FAILED: {type(exc).__name__}: {str(exc)[:140]}")
            continue

        shapes = [np.asarray(out).shape for out in outputs]
        maxima = [float(np.asarray(out).max()) for out in outputs]
        print(f"{name:<32} out={shapes} max={[round(m, 2) for m in maxima]}")


if __name__ == "__main__":
    main()
