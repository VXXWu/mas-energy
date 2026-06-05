"""Diagnose what is constraining jinja2 to <3.1 in the mas-energy env.

Usage (on cluster):
    source ~/.bashrc && conda activate mas-energy
    python mas-energy/scripts/jinja_diag.py
"""
import jinja2
from importlib.metadata import requires, distributions

print("jinja2 version:", jinja2.__version__)
print()
print("Packages declaring a jinja2 constraint:")
found = False
for d in distributions():
    name = d.metadata["Name"]
    for r in (requires(name) or []):
        if "jinja" in r.lower():
            print(f"  {name}: {r}")
            found = True
if not found:
    print("  (none — nothing pins jinja2 explicitly)")
