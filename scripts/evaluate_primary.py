"""Public-data rerun of the primary INT8/FP32 comparison and exploratory R–R comparator.
Portable adaptation, not the historical one-shot execution script. Q15 is not included.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch,yaml
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from prepare_dataset import load_assignments, ROOT
from train_cinc2017_candidate import FrozenNet
from quantize_evaluate_p6_int8 import windows_with_starts,run_int8_windows
from rr_features import extract
SEED=20260921

def paired_bootstrap(labels: np.ndarray, fixed: np.ndarray, fp32: np.ndarray):
    rng = np.random.Generator(np.random.PCG64(SEED))
    valid, attempts, discards = [], 0, {"single_class": 0, "nonfinite": 0, "ap_error": 0}
    while len(valid) < 2000 and attempts < 4000:
        attempts += 1
        ix = rng.integers(0, len(labels), len(labels))
        y, a, b = labels[ix], fixed[ix], fp32[ix]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            discards["nonfinite"] += 1
            continue
        if len(np.unique(y)) != 2:
            discards["single_class"] += 1
            continue
        try:
            delta = float(average_precision_score(y, a, pos_label=1) - average_precision_score(y, b, pos_label=1))
        except (ValueError, FloatingPointError):
            discards["ap_error"] += 1
            continue
        if not np.isfinite(delta):
            discards["nonfinite"] += 1
            continue
        valid.append(delta)
    if len(valid) < 2000:
        return {"attempted": attempts, "valid": len(valid), "discarded": discards, "h1": "indeterminate"}
    lo, hi = np.quantile(valid, [0.025, 0.975], method="linear")
    return {"attempted": attempts, "valid": len(valid), "discarded": discards,
            "ci95_lower": float(lo), "ci95_upper": float(hi), "noninferiority_delta": 0.02,
            "h1": "noninferior" if lo > -0.02 else "noninferiority_not_established"}

def main():
    p=argparse.ArgumentParser();p.add_argument('--ecg-dir',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    assignments=load_assignments(a.reference);test=[r for r in assignments if r['split']=='locked_test'];train=[r for r in assignments if r['split']=='train']
    def path(r):return a.ecg_dir/(r['record_id']+'.mat')
    y=np.array([r['mapped_label'] for r in test]);ty=np.array([r['mapped_label'] for r in train])
    qmeta=json.loads((ROOT/'model/quantization.json').read_text());q=np.load(ROOT/'model/int8_parameters.npz',allow_pickle=False)
    layers=[dict(qw=q[f'w{i}'],qb=q[f'b{i}'],sw=np.asarray(v['weight_scales']),s_in=v['input_activation_scale'],s_out=v['output_activation_scale']) for i,v in enumerate(qmeta['layers'])]
    fp=FrozenNet('small');weights=np.load(ROOT/'model/fp32_weights.npz',allow_pickle=False);fp.load_state_dict({k:torch.from_numpy(weights[k]) for k in weights.files});fp.eval()
    windows=[];owner=[]
    for j,r in enumerate(test):
        for _,_,norm in windows_with_starts(path(r),1500,750,False):windows.append(norm);owner.append(j)
    x=np.stack(windows);owner=np.asarray(owner);counts=np.bincount(owner,minlength=len(test));assert np.all(counts>0)
    a8=run_int8_windows(x,layers,qmeta['activation_scales']);f32=[]
    with torch.inference_mode():
        for i in range(0,len(x),128):f32.append(torch.sigmoid(fp(torch.from_numpy(x[i:i+128,None,:]))).double().numpy())
    def aggregate(s):
        sums=np.zeros(len(test));np.add.at(sums,owner,s);return sums/counts
    cnn=aggregate(a8);ref=aggregate(np.concatenate(f32))
    threshold=yaml.safe_load((ROOT/'model/selected_config.yaml').read_text())['threshold']; pred=cnn>=threshold;positive=y==1
    tp=int(np.sum(pred&positive));fn=int(np.sum(~pred&positive));tn=int(np.sum(~pred&~positive));fpn=int(np.sum(pred&~positive))
    X=np.asarray([extract(path(r))[0] for r in train]);T=np.asarray([extract(path(r))[0] for r in test]);X[~np.isfinite(X)]=np.nan;T[~np.isfinite(T)]=np.nan
    imp=SimpleImputer(strategy='median',keep_empty_features=True);Xi=imp.fit_transform(X)
    rr=HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.,random_state=20260927);rr.fit(Xi,ty);rrs=rr.predict_proba(imp.transform(T))[:,1]
    def metrics(s):return {'AP':float(average_precision_score(y,s)),'AUROC':float(roc_auc_score(y,s))}
    res={'records':len(test),'windows':len(x),'INT8':metrics(cnn),'FP32':metrics(ref),'RR':metrics(rrs),'INT8_minus_FP32_AP':float(average_precision_score(y,cnn)-average_precision_score(y,ref)),'paired_INT8_FP32':paired_bootstrap(y,cnn,ref),'RR_minus_CNN_AP':float(average_precision_score(y,rrs)-average_precision_score(y,cnn)),'RR_minus_CNN_CI':{k:v for k,v in paired_bootstrap(y,rrs,cnn).items() if k in ('ci95_lower','ci95_upper','valid')},'threshold':threshold,'TP':tp,'FN':fn,'TN':tn,'FP':fpn,'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fpn),'F1':2*tp/(2*tp+fn+fpn)}
    print(json.dumps(res,indent=2))
if __name__=='__main__':main()
