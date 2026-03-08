#!/usr/bin/env python3
from __future__ import annotations
import argparse, datetime as dt, json, re, subprocess
from pathlib import Path

DEFAULT_LRS = [5e-7,1e-6,2e-6,3e-6,5e-6,7e-6,1e-5,1.5e-5,2e-5,3e-5]

def fmt_lr(lr: float) -> str: return f"{lr:.10g}"
def slug_lr(lr: float) -> str: return 'lr_' + fmt_lr(lr).replace('+','').replace('-','m').replace('.','p')

def replace_yaml_value(text: str, key: str, value: str) -> str:
    p = re.compile(rf"^(\s*{re.escape(key)}\s*:\s*)([^#\n]*)(.*)$", re.MULTILINE)
    if not p.search(text): raise ValueError(f"Key '{key}' not found")
    return p.sub(rf"\1{value}\3", text, count=1)

def replace_train_line(text: str, yaml_path: str) -> str:
    p = re.compile(r"^\s*llamafactory-cli\s+train\s+.+$", re.MULTILINE)
    line = f"llamafactory-cli train {yaml_path}"
    return p.sub(line, text, count=1) if p.search(text) else text.rstrip()+"\n"+line+"\n"

def submit_sbatch(path: Path):
    proc = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0: return False, out, None
    m = re.search(r"Submitted batch job\s+(\d+)", out)
    return True, out, (m.group(1) if m else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-yaml", default="config.yaml")
    ap.add_argument("--template-slurm", default="run_example.slurm.sh")
    ap.add_argument("--base-output-dir", default="factory_qwen_results")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    yaml_tpl = (root / args.template_yaml).read_text(encoding="utf-8")
    slurm_tpl = (root / args.template_slurm).read_text(encoding="utf-8")

    yamls = root / "yamls"; slurms = root / "slurms"; outs = root / "slurm_outputs"
    for d in (yamls, slurms, outs): d.mkdir(parents=True, exist_ok=True)

    rows = []
    for lr in DEFAULT_LRS:
        lr_s = fmt_lr(lr); tag = slug_lr(lr)
        ypath = yamls / f"{tag}.yaml"
        spath = slurms / f"{tag}.slurm"

        y = replace_yaml_value(yaml_tpl, "learning_rate", lr_s)
        y = replace_yaml_value(y, "output_dir", f"{args.base_output_dir}/{tag}")
        ypath.write_text(y, encoding="utf-8")

        s = replace_train_line(slurm_tpl, str(ypath))
        s = re.sub(r"^#SBATCH --output=.*$", f"#SBATCH --output={outs / (tag + '.out')}", s, flags=re.MULTILINE)
        s = re.sub(r"^#SBATCH --error=.*$", f"#SBATCH --error={outs / (tag + '.err')}", s, flags=re.MULTILINE)
        if not s.startswith("#!/bin/bash"): s = "#!/bin/bash\n" + s
        spath.write_text(s, encoding="utf-8")

        ok, out, jid = (False, "not submitted", None)
        if args.submit: ok, out, jid = submit_sbatch(spath)
        rows.append({"lr": lr_s, "tag": tag, "yaml": str(ypath), "slurm": str(spath), "stdout": str(outs/(tag+'.out')), "stderr": str(outs/(tag+'.err')), "submitted": bool(args.submit), "submit_ok": ok, "job_id": jid, "submit_output": out})

    manifest = root / f"manifest_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest.write_text(json.dumps({"jobs": rows}, indent=2), encoding="utf-8")
    print(f"Created {len(rows)} jobs")
    print(f"Manifest: {manifest}")

if __name__ == '__main__': main()
