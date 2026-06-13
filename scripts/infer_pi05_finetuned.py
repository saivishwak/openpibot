#!/usr/bin/env python3
"""Compatibility CLI for the OpenPiBot PI0.5 inference runtime."""

from __future__ import annotations

from openpibot import pi05_inference_runtime as _runtime

for _name, _value in vars(_runtime).items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__"}:
        globals()[_name] = _value

main = _runtime.main


if __name__ == "__main__":
    main()
