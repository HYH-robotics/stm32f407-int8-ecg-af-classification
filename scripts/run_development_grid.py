"""Portable launcher for the original 12 configurations."""
from pathlib import Path
import subprocess,sys
from quantize_evaluate_p6_int8 import CONFIGS,ROOT,TRAIN_MANIFEST,TRAIN_SPLIT
for window,hop,preprocessing,model,name in CONFIGS:
    subprocess.run([sys.executable,str(ROOT/'scripts/train_cinc2017_candidate.py'),'--manifest',str(TRAIN_MANIFEST),'--split',str(TRAIN_SPLIT),'--run-dir',str(ROOT/'runs'/name),'--window-s',str(window),'--hop-s',str(hop),'--preprocessing',preprocessing,'--model',model,'--seed','20260921'],check=True)
