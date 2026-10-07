## Tests

```sh
python -m pip install -r requirements-test.txt
python -m pytest
```

These initial regression tests cover selected existing behavior. External services are mocked or not invoked; this is not end-to-end coverage.

Coverage includes the existing evals/test_pipeline.py suite plus chunking, metadata isolation, and local password registration/verification. Model and ingestion integrations are not covered.
