# How to run these real integration tests

```bash
# First, export the ONNX model
python scripts/export_model.py \
  --model owlv2-base \
  --output models/owl2_model.onnx \
  --device cpu

# Run all integration tests with verbose output
pytest tests/integration/test_owl.py -v -s

# Run only the shape validation tests
pytest tests/integration/test_owl.py::TestONNXModelOutputShapes -v

# Run only the detection tests
pytest tests/integration/test_owl.py::TestONNXModelDetectsObjects -v

# Run only the service tests
pytest tests/integration/test_owl.py::TestServiceIntegration -v

# Run only the API endpoint tests
pytest tests/integration/test_owl.py::TestAPIEndpoint -v

# Run with custom model path
pytest tests/integration/test_owl.py \
  --model-path /path/to/custom/owl2.onnx \
  -v -s

# Skip integration tests (e.g., in CI/CD without model)
pytest tests/ -m "not integration"
```