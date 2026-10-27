# -*- coding: utf-8 -*-
"""Folders of the internal inputs read by the second-revision scripts.

The inputs carry cell identifiers, coordinates or registered site locations and are not released. Set these
environment variables to the corresponding folders before running a script."""
import os
from pathlib import Path


def _folder(name, default):
    return Path(os.environ.get(name, default))


ROUND1 = _folder("REGAIN_ROUND1_ANALYSIS", "internal/round1_analysis")      # round-1 analysis tables and protocols
ROUND2 = _folder("REGAIN_ROUND2_ANALYSIS", "internal/round2_analysis")      # outputs of the second revision
MODEL_INPUTS = _folder("REGAIN_MODEL_INPUTS", "internal/model_inputs")      # annual climate inputs by cell
VECTORS = _folder("REGAIN_VECTORS", "internal/vectors")                     # site inventory and mapped footprints
BOUNDARY = _folder("REGAIN_BOUNDARY", "internal/boundary")                  # prefecture and county boundaries
