#!/usr/bin/env python3
"""Launcher for running the MDF genetic algorithm with the SMC-DEMC pipeline."""

import runpy
import sys

if __name__ == "__main__":
    argv = sys.argv[:]

    if not any(arg.startswith("--plot-mode") for arg in argv[1:]):
        argv = [argv[0], "--plot-mode", "posterior_minimal", *argv[1:]]

    sys.argv = argv
    runpy.run_module("MDF_GA", run_name="__main__")
