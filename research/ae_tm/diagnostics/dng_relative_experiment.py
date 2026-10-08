"""Real-content relative exposure diagnostic from local DNG green patches.

All backend/policy selection is train/development-only. The fixed target is
derived from an already bounded/noisy processed DNG, not expert GT or HDR.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from scipy.ndimage import uniform_filter
from continuous_reference import (ADCLikelihood, ADC_MAX, FW, READ_SD,
                                  EXPOSURES, capture_codes, tone)

FEATURE_NAMES=("mean_brightness","q99_brightness","temporal_noise","local_gradient")
WINDOWS=(1,3,5,7)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_patch(cache, source):
    with np.load(cache/f"{source['source_id']}.npz",allow_pickle=False) as data:
        x=data["green"].astype(np.float64)
    if x.ndim!=3 or not np.isfinite(x).all() or np.min(x)<0 or np.max(x)>1:
        raise ValueError("Expected finite bounded green patches")
    return x


def build_prior(cache,sources,camera,bins=256,pseudomass=1e-6):
    """Group-balanced train density with endpoint atoms; no diagnostic pixels."""
    grouped={}
    for source in sources:
        if source["camera"]!=camera or source["split"]!="train":
            continue
        x=load_patch(cache,source).ravel()
        edges=np.linspace(0,1,bins+1)
        hist,_=np.histogram(x[(x>0)&(x<1)],bins=edges)
        mass=np.r_[np.mean(x==0),hist/x.size,np.mean(x==1)]
        grouped.setdefault(source["group_id"],[]).append(mass)
    if not grouped:
        raise ValueError("No training source groups")
    mass=np.mean([np.mean(items,axis=0) for items in grouped.values()],axis=0)
    # Fixed tiny continuous-density floor ensures numerical evidence at rare ADCs.
    mass[1:-1]+=pseudomass/bins
    mass/=mass.sum()
    return mass,sorted(grouped)


def prior_nodes(mass,nodes_per_bin):
    bins=len(mass)-2
    u,w=np.polynomial.legendre.leggauss(nodes_per_bin)
    values=((np.arange(bins)[:,None]+(u+1)/2)/bins).ravel()
    weights=(mass[1:-1,None]*w[None,:]/2).ravel()
    return np.r_[0.,values,1.],np.r_[mass[0],weights,mass[-1]]


def posterior_tables(mass,nodes_per_bin,likelihood):
    """Exact ADC-code lookup of quadrature posterior; no code bin merging."""
    x,weights=prior_nodes(mass,nodes_per_bin)
    target=tone(x)
    tables=[]
    for exposure in EXPOSURES:
        p=likelihood.probabilities(np.arange(ADC_MAX+1),FW*exposure*x)
        evidence=p@weights
        if np.any(evidence<=0) or not np.isfinite(evidence).all():
            raise FloatingPointError("Zero/nonfinite point posterior evidence")
        tables.append((p@(weights*target))/evidence)
    return np.array(tables)


def wiener_target(codes,exposure,window):
    signal=codes.astype(np.float64)/ADC_MAX/exposure
    if window==1:
        return tone(np.clip(signal,0,1))
    shape=(1,1,window,window)
    mean=uniform_filter(signal,size=shape,mode="reflect")
    variance=np.maximum(0,uniform_filter(signal*signal,size=shape,mode="reflect")-mean*mean)
    noise=(np.maximum(mean,0)/(FW*exposure)
           +(READ_SD/(FW*exposure))**2+1/(12*ADC_MAX**2*exposure**2))
    gain=np.clip(1-noise/np.maximum(variance,1e-20),0,1)
    estimate=mean+gain*(signal-mean)
    return tone(np.clip(estimate,0,1))


def preview_features(x,rng):
    previews=capture_codes(np.broadcast_to(x,(3,)+x.shape),.25,rng)
    values=previews.astype(np.float64)/ADC_MAX/.25
    mean=values.mean(axis=0)
    gradient=np.concatenate((np.diff(mean,axis=1).ravel(),np.diff(mean,axis=2).ravel()))
    return np.array([mean.mean(),np.quantile(mean,.99),
                     np.sqrt(values.var(axis=0,ddof=1).mean()),
                     np.mean(np.abs(gradient))])


def source_risks(x,repeats,rng,tables,tables_low):
    target=tone(x)[None,...]
    inverse=np.empty(len(EXPOSURES))
    empirical=np.empty_like(inverse)
    empirical_low=np.empty_like(inverse)
    spatial=np.empty((len(WINDOWS),len(EXPOSURES)))
    repeat_se=np.empty((3,len(EXPOSURES)))
    for index,exposure in enumerate(EXPOSURES):
        codes=capture_codes(np.broadcast_to(x,(repeats,)+x.shape),exposure,rng)
        predictions=[
            tone(np.clip(codes.astype(float)/ADC_MAX/exposure,0,1)),
            tables[index,codes],tables_low[index,codes]]
        for model,prediction in enumerate(predictions):
            losses=((prediction-target)**2).mean(axis=(1,2,3))
            if model==0:inverse[index]=losses.mean()
            elif model==1:empirical[index]=losses.mean()
            else:empirical_low[index]=losses.mean()
            repeat_se[model,index]=losses.std(ddof=1)/np.sqrt(repeats)
        for widx,window in enumerate(WINDOWS):
            prediction=wiener_target(codes,exposure,window)
            spatial[widx,index]=((prediction-target)**2).mean()
    return {"inverse":inverse,"point_posterior":empirical,
            "point_posterior_low":empirical_low,"spatial_windows":spatial,
            "future_noise_mean_se":repeat_se}


def group_mean(values,records):
    groups=sorted({row["group_id"] for row in records})
    return np.mean([np.mean(values[[i for i,row in enumerate(records)
                                     if row["group_id"]==g]],axis=0) for g in groups],axis=0)


def choose_rule(train_features,dev_features,costs,dev_records,feature):
    """Same two-action threshold capacity; group-balanced dev risk."""
    best=None
    candidates=np.unique(np.quantile(train_features[:,feature],np.arange(.1,1,.1)))
    for threshold in candidates:
        left=dev_features[:,feature]<=threshold
        if left.all() or not left.any():
            continue
        for a in range(len(EXPOSURES)):
            for b in range(len(EXPOSURES)):
                selected=np.where(left,costs[:,a],costs[:,b])
                score=float(group_mean(selected,dev_records))
                if best is None or score<best["dev_group_balanced_risk"]:
                    best={"feature":FEATURE_NAMES[feature],"threshold":float(threshold),
                          "left":a,"right":b,"dev_group_balanced_risk":score,
                          "candidate_thresholds_train_only":candidates.tolist()}
    if best is None:
        a=int(group_mean(costs,dev_records).argmin())
        best={"feature":FEATURE_NAMES[feature],"threshold":float(np.median(train_features[:,feature])),
              "left":a,"right":a,"dev_group_balanced_risk":float(group_mean(costs[:,a],dev_records)),
              "degenerate":"same fixed action"}
    return best


def policy_actions(features,rule):
    index=FEATURE_NAMES.index(rule["feature"])
    return np.where(features[:,index]<=rule["threshold"],rule["left"],rule["right"])


def summarize_diagnostic(costs,features,records,fixed,rules,rng):
    fixed_cost=costs[:,fixed]
    groups=sorted({row["group_id"] for row in records})
    result={"dev_fixed_exposure":float(EXPOSURES[fixed]),
            "group_balanced_risk_by_action":group_mean(costs,records).tolist(),
            "fixed_risk":float(group_mean(fixed_cost,records)),
            "diagnostic_group_count":len(groups),"policies":{}}
    for name,rule in rules.items():
        actions=policy_actions(features,rule)
        values=costs[np.arange(len(costs)),actions]
        diffs=np.array([np.mean((values-fixed_cost)[[i for i,row in enumerate(records)
                                                  if row["group_id"]==g]]) for g in groups])
        sampled=diffs[rng.integers(0,len(diffs),(2000,len(diffs)))].mean(axis=1)
        risk=float(group_mean(values,records))
        result["policies"][name]={"risk":risk,
            "absolute_difference_policy_minus_fixed":float(diffs.mean()),
            "relative_reduction_vs_fixed":float(1-risk/result["fixed_risk"]),
            "descriptive_group_bootstrap_ci95":np.quantile(sampled,[.025,.975]).tolist(),
            "source_action_counts":np.bincount(actions,minlength=len(EXPOSURES)).tolist()}
    return result


def run(cache,audit_path,output,seeds=(19,37,73),repeats=4,bins=256,
        nodes_per_bin=4,low_nodes_per_bin=2):
    begin=time.perf_counter()
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    sources=audit["sources"]
    cache_manifest=json.loads((cache/"cache_manifest.json").read_text(encoding="utf-8"))
    if cache_manifest["protocol_sha256"]!=audit["protocol_sha256"]:
        raise ValueError("Cache/protocol mismatch")
    output.mkdir(parents=True,exist_ok=False)
    likelihood=ADCLikelihood()
    report={"protocol_id":audit["protocol_id"],"protocol_sha256":audit["protocol_sha256"],
            "source_manifest_sha256":audit["manifest_sha256"],
            "cache_sha256":{source["source_id"]:file_sha(cache/f"{source['source_id']}.npz") for source in sources},
            "data_domain":audit["source_domain"],
            "independence_limit":audit["independence_limit"],
            "status":"completed_real_content_relative_diagnostic",
            "target":"sqrt(x/(x+.5)) of bounded/noisy green DNG source",
            "actions":EXPOSURES.tolist(),
            "sensor_assumptions":{"full_well":FW,"read_sd":READ_SD,"adc_max":ADC_MAX,
                                  "camera_calibrated":False,"clip_after_read":True},
            "config":{"seeds":list(seeds),"future_repeats":repeats,"prior_bins":bins,
                      "nodes_per_bin":nodes_per_bin,"low_nodes_per_bin":low_nodes_per_bin,
                      "prior_pseudomass":1e-6,"spatial_windows":list(WINDOWS)},
            "selection":"camera-specific train prior, group-balanced development selects windows/rules/fixed; then freezes",
            "backend_limits":"optimal point posterior only within estimated 1D prior; simple local Wiener is not unrestricted strong image TM",
            "software":{"python":platform.python_version(),"numpy":np.__version__},
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "likelihood_audit":likelihood.audit(),"cameras":{},"replicates":[]}
    tables={}
    for camera in ("iphone","s25"):
        mass,train_groups=build_prior(cache,sources,camera,bins)
        tables[camera]=(posterior_tables(mass,nodes_per_bin,likelihood),
                        posterior_tables(mass,low_nodes_per_bin,likelihood))
        report["cameras"][camera]={"train_groups":train_groups,
                                  "group_balanced_prior_mass":mass.tolist(),
                                  "endpoint_mass":[float(mass[0]),float(mass[-1])]}
    metric_rows=[]
    for seed in seeds:
        records=[]
        for source in sources:
            x=load_patch(cache,source)
            cam_id=0 if source["camera"]=="iphone" else 1
            streams=np.random.SeedSequence([seed,source["pair_id"],cam_id]).spawn(2)
            features=preview_features(x,np.random.default_rng(streams[0]))
            # Control is full continuous, unclipped, noiseless inverse, outside ADC model.
            invariant=max(float(np.max(np.abs(tone(x*e/e)-tone(x)))) for e in EXPOSURES)
            if invariant>1e-14:
                raise AssertionError("Noiseless scale inversion invariant failed")
            risks=source_risks(x,repeats,np.random.default_rng(streams[1]),
                               *tables[source["camera"]])
            records.append({**{k:source[k] for k in
                               ("source_id","camera","pair_id","group_id","split")},
                            "features":features,"risks":risks})
        replicate={"seed":seed,"cameras":{}}
        for cam_id,camera in enumerate(("iphone","s25")):
            sets={split:[r for r in records if r["camera"]==camera and r["split"]==split]
                  for split in ("train","development","diagnostic")}
            train,dev,diag=(sets[n] for n in ("train","development","diagnostic"))
            tf=np.stack([r["features"] for r in train])
            df=np.stack([r["features"] for r in dev])
            vf=np.stack([r["features"] for r in diag])
            spatial_dev=np.stack([r["risks"]["spatial_windows"] for r in dev])
            window_risks=group_mean(spatial_dev,dev)
            window_indices=window_risks.argmin(axis=0)
            def costs_for(rows,backend):
                if backend=="local_wiener":
                    return np.stack([r["risks"]["spatial_windows"][window_indices,np.arange(len(EXPOSURES))]
                                     for r in rows])
                return np.stack([r["risks"][backend] for r in rows])
            cresult={"dev_selected_spatial_windows":[WINDOWS[i] for i in window_indices],
                     "backends":{}}
            for backend in ("inverse","point_posterior","local_wiener"):
                dc,vc=costs_for(dev,backend),costs_for(diag,backend)
                fixed=int(group_mean(dc,dev).argmin())
                rules={name:choose_rule(tf,df,dc,dev,i) for i,name in enumerate(FEATURE_NAMES)}
                cresult["backends"][backend]={
                    "dev_risk_by_action":group_mean(dc,dev).tolist(),
                    "frozen_rules":rules,
                    "diagnostic":summarize_diagnostic(vc,vf,diag,fixed,rules,
                        np.random.default_rng(np.random.SeedSequence([seed,cam_id,773])) )}
                for i,row in enumerate(diag):
                    actions={name:int(policy_actions(vf,rule)[i]) for name,rule in rules.items()}
                    metric_rows.append({"seed":seed,"camera":camera,"backend":backend,
                        "source_id":row["source_id"],"group_id":row["group_id"],"split":row["split"],
                        "dev_fixed_action":fixed,
                        **{f"risk_e{e:g}":float(vc[i,j]) for j,e in enumerate(EXPOSURES)},
                        **{f"policy_action_{n}":v for n,v in actions.items()}})
            high=costs_for(diag,"point_posterior")
            low=costs_for(diag,"point_posterior_low")
            cresult["quadrature_convergence"]={
                "low_minus_high_risk_by_action":group_mean(low-high,diag).tolist(),
                "max_absolute_group_action_risk_difference":
                    float(np.max(np.abs(group_mean(low-high,diag))))}
            replicate["cameras"][camera]=cresult
        report["replicates"].append(replicate)
        print(json.dumps({"seed":seed,"sources":len(records),"status":"done"}),flush=True)
    with (output/"source_metrics.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    report["source_metrics_sha256"]=file_sha(output/"source_metrics.csv")
    report["elapsed_seconds"]=time.perf_counter()-begin
    report["limitations"]=[
        "one capture session, 11 conservative background groups; only3 diagnostic groups",
        "cropping8patches per image and noise replicas do not increase independent scene count",
        "source already processed/noisy/clipped; no expert GT or actual recapture",
        "simulated full-well/read noise are assumptions, not per-camera calibration",
        "point prior and local Wiener are limited backend families; no universal strong-TM conclusion",
        "camera-specific windows/rules selected on only2 development groups; selection uncertainty",
        "bootstrap intervals are descriptive and conditional on frozen development selections",
        "no original Apple/Samsung catalog reproduction, motion, Bayer or formal S24 test"
    ]
    (output/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",required=True,type=Path)
    parser.add_argument("--audit",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--seeds",nargs="+",type=int,default=[19,37,73])
    parser.add_argument("--repeats",type=int,default=4)
    args=parser.parse_args()
    if args.repeats<2:
        parser.error("Need >=2 future repetitions")
    result=run(args.cache,args.audit,args.out,args.seeds,args.repeats)
    print(json.dumps({"status":result["status"],"elapsed_seconds":result["elapsed_seconds"]}))
