# OWL 2 Object Detection Inference Service

## OWL v2 App Structure

```
owl/
├── inference/
│   ├── __init__.py
│   ├── router.py        # All endpoints for this domain
│   ├── schemas.py       # Pydantic request/response models
│   ├── service.py       # ONNX inference business logic
│   ├── dependencies.py  # Shared injectable dependencies
│   ├── constants.py     # Error codes, config keys
│   ├── exceptions.py    # Domain-specific exceptions
│   └── config.py        # Domain-scoped settings
├── config.py            # Global owl app settings
├── exceptions.py        # Global owl exception handlers
├── main.py              # App entrypoint
└── models/
    └── owl2_model.onnx      # Exported OWL 2 ONNX model
```

## For Developers

The following instructions are for developers who wish build their own inference pipeline using this repository's
components.

### Exporting the models

Before exporting the `OWL` or any other supported object detection framework we need first install all dependencies:

```shell
pip install -r requirements/export.txt
```

#### Downloading and exporting OWL-2

To download and export the owl-2 use the following CLI script:

```bash
# Export owlv2-base to ONNX with optimization
python scripts/export_model.py \
  --model owlv2-base \
  --output src/owl/models/cuda/owl_model.onnx \
  --device cuda \
  --optimize

# Export owlv2-large without optimization
python scripts/export_model.py \
  --model owlv2-large \
  --output src/owl/models/cpu/owl_large.onnx \
  --device cpu \
  --no-optimize         
```

Alternatively, you can also run a Programmatic API:

```Python
from pathlib import Path
from src.owl.inference.utils import export_owl2_to_onnx

# One-liner export
onnx_path = export_owl2_to_onnx(
    model_name="owlv2-base",
    output_path=Path("src/owl/models/owl2_model.onnx"),
    device="cuda",
    optimize=True,
)

print(f"Model exported to: {onnx_path}")
```

If you want full control over your export process, use the following code snippet:

```python
from pathlib import Path
from src.owl.inference.utils import OWL2ModelExporter

exporter = OWL2ModelExporter(
    model_name="owlv2-base",
    device="cuda",
)

exporter.load_model()

exporter.export_to_onnx(
    output_path=Path("src/owl/models/owl2_model.onnx"),
    opset_version=14,
    optimize_model=True,
    use_external_data_format=False,  # For very large models
)
```

##### Running the service

```bash
# Install dependencies
pip install -r requirements/base.txt

# Start with uvicorn (single worker for dev)
uvicorn src.owl.main:app --reload --host 0.0.0.0 --port 8000

# Production: use multiple workers via gunicorn
gunicorn src.owl.main:app \
  -k uvicorn.workers.UvicornWorker \
  -w 4 \
  --bind 0.0.0.0:8000
```

##### How to build and run it locally

```bash
# Build the image
docker build -t owl2-inference:latest .

# Run with docker-compose (recommended for local dev)
docker-compose up --build

# Or run standalone
docker run -it \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e ENVIRONMENT=local \
  owl2-inference:latest

# Run the export image to create ONNX model
docker build -f Dockerfile.export -t owl2-export:latest .

docker run --rm \
  -v $(pwd)/models:/app/models \
  owl2-export:latest \
  --model owlv2-base \
  --output /app/models/owl2_model.onnx \
  --device cpu \
  --optimize
```