## dino Structure

```
dino/
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
└── dino_model.onnx      # Exported OWL 2 ONNX model
```