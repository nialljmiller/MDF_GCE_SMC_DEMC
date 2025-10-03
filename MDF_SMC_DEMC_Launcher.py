#!/usr/bin/env python3
"""Launcher for running the MDF genetic algorithm with the SMC-DEMC pipeline."""

import runpy
import sys

if __name__ == "__main__":
    # Pass command-line arguments straight through to MDF_GA
    sys.argv = sys.argv[:]
    runpy.run_module("MDF_GA", run_name="__main__")
