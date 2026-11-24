"""Recompute timing summaries from anonymous cycle counts; no ECG data needed."""
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
root=Path(__file__).resolve().parents[1];groups=defaultdict(list)
for name,keys in [('cmsis_comparison_cycles.csv',('implementation','batch')),('model_scale_cycles.csv',('configuration',))]:
    with (root/'board'/name).open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):groups['/'.join(r[k] for k in keys)].append(int(r['cycles']))
def stats(c):
    t=np.array(c,dtype=float)/168000
    return dict(n=len(t),mean_ms=float(t.mean()),sd_ms=float(t.std(ddof=1)),median_ms=float(np.quantile(t,.5)),p95_ms=float(np.quantile(t,.95)),p99_ms=float(np.quantile(t,.99)),max_ms=float(t.max()),deadline_misses=int(np.sum(t>2500)))
print(json.dumps({k:stats(v) for k,v in groups.items()},indent=2))
