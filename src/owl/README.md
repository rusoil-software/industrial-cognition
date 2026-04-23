
## owl Structure

```
fastapi-project/
├── src/
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── router.py        # All endpoints for this domain
│   │   ├── schemas.py       # Pydantic request/response models
│   │   ├── service.py       # ONNX inference business logic
│   │   ├── dependencies.py  # Shared injectable dependencies
│   │   ├── constants.py     # Error codes, config keys
│   │   ├── exceptions.py    # Domain-specific exceptions
│   │   └── config.py        # Domain-scoped settings
│   ├── config.py            # Global app settings
│   ├── exceptions.py        # Global exception handlers
│   └── main.py              # App entrypoint
├── models/
│   └── owl2_model.onnx      # Your exported OWL 2 ONNX model
├── tests/
│   └── inference/
│       └── test_router.py
├── requirements/
│   ├── base.txt
│   └── dev.txt
└── .env
```