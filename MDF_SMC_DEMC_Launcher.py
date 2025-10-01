#!/usr/bin/env python3
"""Launcher for running the MDF genetic algorithm with the SMC-DEMC pipeline."""
import runpy

if __name__ == "__main__":
    runpy.run_module("MDF_GA", run_name="__main__")
