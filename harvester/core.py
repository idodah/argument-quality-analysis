"""Canonical name for the shared generation entrypoint.

The rebuttal generator itself lives in `agents.generate` and is reused as-is;
this module only re-exports it so the harvester has a stable internal handle
(`harvester.core.generate_pro_israel_response`) independent of where the graph
happens to live. Do NOT redefine the entrypoint here — import and re-export only.
"""

from __future__ import annotations

from agents.generate import generate_pro_israel_response

__all__ = ["generate_pro_israel_response"]
