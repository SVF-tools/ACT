# ACT Serialization

`serialization.py` provides `NetSerializer` plus `save_net_to_file`, `load_net_from_file`, `save_net_to_string` and `load_net_from_string` for round-tripping a `Net` (including tensors) through JSON.

## Tests

`test_serialization.py` is an assert-based suite; run it from the project root:

```bash
python act/back_end/serialization/test_serialization.py
```

It covers basic round-trip, file I/O, device migration, schema validation, conversion-metadata round-trip and direct VNNLIB export round-trip. The exit code is non-zero on failure.
