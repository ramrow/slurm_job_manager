#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import wandb

def safe_get(d: dict[str, Any], *keys: str, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entity', required=True)
    ap.add_argument('--project', required=True)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--out', default='wandb_lr_summary.json')
    args = ap.parse_args()

    jobs = json.loads(args.manifest.read_text(encoding='utf-8')).get('jobs', [])
    by_lr = {str(j['lr']): {"lr": str(j['lr']), "tag": j.get('tag'), "job_id": j.get('job_id'), "wandb_run": None, "state": None, "summary": {}} for j in jobs}

    api = wandb.Api(timeout=60)
    for run in api.runs(f"{args.entity}/{args.project}"):
        lr = safe_get(run.config, 'learning_rate', default=None)
        if lr is None: continue
        key = f"{float(lr):.10g}" if isinstance(lr, (int,float)) else str(lr)
        if key not in by_lr: continue
        rec = by_lr[key]
        rec['wandb_run'] = run.name
        rec['state'] = run.state
        s = run.summary._json_dict if hasattr(run.summary, '_json_dict') else dict(run.summary)
        rec['summary'] = {
            'train_loss': s.get('train/loss') or s.get('loss'),
            'eval_loss': s.get('eval/loss') or s.get('eval_loss'),
            'train_runtime': s.get('train/runtime'),
            'tokens_per_second': s.get('train/tokens_per_second') or s.get('tokens_per_second'),
            'best_metric': s.get('best_metric') or s.get('eval/loss'),
        }

    rows = sorted(by_lr.values(), key=lambda r: float(r['lr']))
    out = args.manifest.parent / args.out
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding='utf-8')
    print(f"Wrote: {out}")

if __name__ == '__main__': main()
