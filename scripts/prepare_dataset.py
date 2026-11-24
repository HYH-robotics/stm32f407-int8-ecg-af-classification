"""Create local train/development inputs from the original partition IDs and public ECG data."""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/split_manifest.csv"
EXPECTED_LABEL_HASH = "92fe220a6310ddf807bd9abf2f589a3202888d7b717689612839b54e6d657dbd"
EXPECTED = {"train": (5774, 530), "development": (825, 76), "pilot": (825, 76), "locked_test": (825, 76)}

def load_assignments(reference):
    with Path(reference).open(encoding="utf-8-sig", newline="") as f:
        raw = {r[0].strip().replace('\\', '/').rsplit('/', 1)[-1]:r[1].strip() for r in csv.reader(f) if len(r)>=2 and r[1].strip() in {"A","N","O","~"}}
    with SPLIT.open(newline="",encoding="utf-8") as f: rows = list(csv.DictReader(f))
    assert len(rows)==8249 and len({r['record_id'] for r in rows})==8249
    assignments=[]
    for r in rows:
        rid=r['record_id']; label=raw.get(rid)
        if label not in {'A','N','O'}: raise ValueError('Reference labels do not match the eligible split')
        assignments.append(dict(record_id=rid,mapped_label=int(label=='A'),split=r['partition']))
    check=hashlib.sha256(''.join(f"{r['record_id']},{r['mapped_label']}\n" for r in assignments).encode()).hexdigest()
    if check!=EXPECTED_LABEL_HASH: raise ValueError('Reference binary labels differ from the original study mapping; select the matching official REFERENCE file.')
    for part,(n,pos) in EXPECTED.items():
        sub=[r for r in assignments if r['split']==part]
        assert len(sub)==n and sum(r['mapped_label'] for r in sub)==pos
    return assignments

def main():
    p=argparse.ArgumentParser();p.add_argument('--ecg-dir',type=Path,required=True,help='Directory containing Axxxxx.mat files');p.add_argument('--reference',type=Path,required=True,help='Official reference CSV');a=p.parse_args()
    assignments=load_assignments(a.reference)
    view=[r for r in assignments if r['split'] in ('train','development')]
    root=a.ecg_dir.resolve()
    records=[]
    for r in view:
        path=root/(r['record_id']+'.mat')
        if not path.is_file(): raise FileNotFoundError('Missing required training/development .mat file')
        records.append(dict(record_id=r['record_id'],mapped_label=r['mapped_label'],relative_path=path.name,source_fs_hz=300,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    split_hash=hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    # Local portable manifest identity; not the historical full-corpus manifest digest.
    meta=dict(view_scope=['train','development'],parent_manifest_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),parent_split_sha256=split_hash)
    dest=ROOT/'data/ecg/views/train_dev_v2';dest.mkdir(parents=True,exist_ok=True)
    (dest/'train_development_manifest.yaml').write_text(yaml.safe_dump(dict(**meta,record_root=str(root),records=records),sort_keys=False),encoding='utf-8')
    (dest/'train_development_split.yaml').write_text(yaml.safe_dump(dict(**meta,assignments=view),sort_keys=False),encoding='utf-8')
    print('Prepared 5,774 training and 825 development records. No pilot/test waveform read.')
if __name__=='__main__': main()
