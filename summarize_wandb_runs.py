#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import wandb


def to_lr_key(v):
    try:
        return "{:.10g}".format(float(v))
    except Exception:
        return str(v)


def main():
    ap = argparse.ArgumentParser(description="Summarize W&B runs for LR sweep")
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", default="wandb_lr_summary.json")
    args = ap.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(str(args.manifest))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])

    by_lr = {}
    for j in jobs:
        k = to_lr_key(j.get("lr"))
        by_lr[k] = {
            "lr": k,
            "tag": j.get("tag"),
            "job_id": j.get("job_id"),
            "wandb_run": None,
            "state": None,
            "summary": {},
        }

    api = wandb.Api(timeout=60)
    runs = api.runs("{}/{}".format(args.entity, args.project))

    for run in runs:
        lr = run.config.get("learning_rate", None)
        if lr is None:
            continue
        k = to_lr_key(lr)
        if k not in by_lr:
            continue

        rec = by_lr[k]
        rec["wandb_run"] = run.name
        rec["state"] = run.state

        s = run.summary._json_dict if hasattr(run.summary, "_json_dict") else dict(run.summary)
        rec["summary"] = {
            "train_loss": s.get("train/loss") or s.get("loss"),
            "eval_loss": s.get("eval/loss") or s.get("eval_loss"),
            "train_runtime": s.get("train/runtime"),
            "tokens_per_second": s.get("train/tokens_per_second") or s.get("tokens_per_second"),
            "best_metric": s.get("best_metric") or s.get("eval/loss"),
        }

    rows = list(by_lr.values())
    rows.sort(key=lambda r: float(r["lr"]))

    out_path = args.manifest.parent / args.out
    out_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    print("Wrote: {}".format(out_path))
    print("\nLR | run | state | eval_loss | train_loss")
    print("-" * 72)
    for r in rows:
        s = r.get("summary", {})
        print("{lr:>8} | {run:18} | {state:8} | {ev} | {tr}".format(
            lr=r["lr"],
            run=str(r.get("wandb_run"))[:18],
            state=str(r.get("state")),
            ev=s.get("eval_loss"),
            tr=s.get("train_loss"),
        ))


if __name__ == "__main__":
    main()
