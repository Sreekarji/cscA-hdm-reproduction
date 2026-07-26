import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
for sub in ["code/hdm", "code", "code/experiments"]:
    p = os.path.join(BASE, sub)
    if os.path.isdir(p):
        sys.path.insert(0, p)

from final_results import run_ablation_N
run_ablation_N()
