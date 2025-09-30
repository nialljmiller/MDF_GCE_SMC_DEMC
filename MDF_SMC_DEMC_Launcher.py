#!/usr/bin/env python3
"""Launcher for the MDF GCE pipeline with SMC-DEMC posterior refinement.

The launcher enforces the reduced plotting bundle tailored to the posterior-focused
SMC-DEMC workflow.  Pass ``--plot-mode`` explicitly when invoking the launcher to
override this default.
"""
import runpy
import sys

if __name__ == "__main__":
    argv = sys.argv[:]

    if not any(arg.startswith("--plot-mode") for arg in argv[1:]):
        argv = [argv[0], "--plot-mode", "posterior_minimal", *argv[1:]]

    sys.argv = argv
    runpy.run_module("MDF_GA", run_name="__main__")
