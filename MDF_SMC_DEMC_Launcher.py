#!/usr/bin/env python3
"""Launcher for the MDF GCE pipeline with SMC-DEMC posterior refinement."""
import runpy

if __name__ == "__main__":
    runpy.run_module("MDF_GA", run_name="__main__")
