#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

DEFAULT_LRS = [
    0.0006, 0.0007, 0.0008, 0.001, 0.0012,
    0.0015, 0.002, 0.00075, 0.0009, 0.0011,
]

DEFAULT_WARMUP_STEPS = [
    75, 100, 120, 150, 80, 60, 100, 200, 125, 180,
]


def fmt_lr(lr):
    return "{:.10g}".format(lr)


def slug_lr(lr):
    return "lr_" + fmt_lr(lr).replace("+", "").replace("-", "m").replace(".", "p")


def slug_job(lr, warmup):
    return "{}_wu_{}".format(slug_lr(lr), int(warmup))


def replace_yaml_value(text, key, value):
    p = re.compile(r"^(\s*" + re.escape(key) + r"\s*:\s*)([^#\n]*)(.*)$", re.MULTILINE)
    if not p.search(text):
        raise ValueError("Key '{}' not found".format(key))
    return p.sub(lambda m: "{}{}{}".format(m.group(1), value, m.group(3)), text, count=1)


def replace_train_line(text, yaml_path):
    p = re.compile(r"^\s*llamafactory-cli\s+train\s+.+$", re.MULTILINE)
    line = "llamafactory-cli train {}".format(yaml_path)
    if p.search(text):
        return p.sub(lambda _m: line, text, count=1)
    return text.rstrip() + "\n" + line + "\n"


def submit_sbatch(path):
    proc = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return False, out, None
    m = re.search(r"Submitted batch job\s+(\d+)", out)
    return True, out, (m.group(1) if m else None)


def main():
    ap = argparse.ArgumentParser(description="Generate + submit LR+warmup sweep jobs")
    ap.add_argument("--template-yaml", default="config.yaml")
    ap.add_argument("--template-slurm", default="run_example.sh")
    ap.add_argument("--base-output-dir", default="output")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    yaml_tpl_path = (root / args.template_yaml).resolve()
    slurm_tpl_path = (root / args.template_slurm).resolve()

    if not yaml_tpl_path.exists():
        raise FileNotFoundError(str(yaml_tpl_path))
    if not slurm_tpl_path.exists():
        raise FileNotFoundError(str(slurm_tpl_path))

    yaml_tpl = yaml_tpl_path.read_text(encoding="utf-8")
    slurm_tpl = slurm_tpl_path.read_text(encoding="utf-8")

    yamls = root / "yamls"
    slurms = root / "slurms"
    outs = root / "output"
    for d in (yamls, slurms, outs):
        d.mkdir(parents=True, exist_ok=True)

    if len(DEFAULT_LRS) != len(DEFAULT_WARMUP_STEPS):
        raise ValueError("DEFAULT_LRS and DEFAULT_WARMUP_STEPS must have same length")

    rows = []
    for lr, warmup in zip(DEFAULT_LRS, DEFAULT_WARMUP_STEPS):
        lr_s = fmt_lr(lr)
        warmup_s = str(int(warmup))
        tag = slug_job(lr, warmup)

        ypath = yamls / (tag + ".yaml")
        spath = slurms / (tag + ".slurm")

        y = replace_yaml_value(yaml_tpl, "learning_rate", lr_s)
        y = replace_yaml_value(y, "warmup_steps", warmup_s)
        y = replace_yaml_value(y, "output_dir", "output/{}".format(tag))
        ypath.write_text(y, encoding="utf-8")

        s = replace_train_line(slurm_tpl, str(ypath))
        s = re.sub(r"^#SBATCH --output=.*$", lambda _m: "#SBATCH --output={}".format(outs / (tag + ".out")), s, flags=re.MULTILINE)
        s = re.sub(r"^#SBATCH --error=.*$", lambda _m: "#SBATCH --error={}".format(outs / (tag + ".err")), s, flags=re.MULTILINE)
        if not s.startswith("#!/bin/bash"):
            s = "#!/bin/bash\n" + s
        spath.write_text(s, encoding="utf-8")

        ok, submit_output, job_id = False, "not submitted", None
        if args.submit:
            ok, submit_output, job_id = submit_sbatch(spath)

        rows.append({
            "lr": lr_s,
            "warmup_steps": int(warmup),
            "tag": tag,
            "yaml": str(ypath),
            "slurm": str(spath),
            "stdout": str(outs / (tag + ".out")),
            "stderr": str(outs / (tag + ".err")),
            "submitted": bool(args.submit),
            "submit_ok": ok,
            "job_id": job_id,
            "submit_output": submit_output,
        })

    manifest = root / ("manifest_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    manifest.write_text(json.dumps({"jobs": rows}, indent=2), encoding="utf-8")
    print("Created {} jobs".format(len(rows)))
    print("Manifest: {}".format(manifest))


if __name__ == "__main__":
    main()


