#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path

DEFAULT_LRS = [5e-7, 1e-6, 2e-6, 3e-6, 5e-6, 7e-6, 1e-5, 1.5e-5, 2e-5, 3e-5]


def fmt_lr(lr):
    return "{:.10g}".format(lr)


def slug_lr(lr):
    return "lr_" + fmt_lr(lr).replace("+", "").replace("-", "m").replace(".", "p")


def replace_yaml_value(text, key, value):
    p = re.compile(r"^(\s*" + re.escape(key) + r"\s*:\s*)([^#\n]*)(.*)$", re.MULTILINE)
    if not p.search(text):
        raise ValueError("Key '{}' not found".format(key))
    return p.sub(lambda m: "{}{}{}".format(m.group(1), value, m.group(3)), text, count=1)


def replace_train_line(text, yaml_path):
    p = re.compile(r"^\s*llamafactory-cli\s+train\s+.+$", re.MULTILINE)
    line = "llamafactory-cli train {}".format(yaml_path)
    if p.search(text):
        return p.sub(line, text, count=1)
    return text.rstrip() + "\n" + line + "\n"


def submit_sbatch(path):
    proc = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        return False, out, None
    m = re.search(r"Submitted batch job\s+(\d+)", out)
    return True, out, (m.group(1) if m else None)


def main():
    ap = argparse.ArgumentParser(description="Generate + submit LR sweep jobs")
    ap.add_argument("--template-yaml", default="config.yaml")
    ap.add_argument("--template-slurm", default="run_example.slurm.sh")
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

    rows = []
    for lr in DEFAULT_LRS:
        lr_s = fmt_lr(lr)
        tag = slug_lr(lr)

        ypath = yamls / (tag + ".yaml")
        spath = slurms / (tag + ".slurm")

        y = replace_yaml_value(yaml_tpl, "learning_rate", lr_s)
        y = replace_yaml_value(y, "output_dir", "output/{}".format(tag))
        ypath.write_text(y, encoding="utf-8")

        s = replace_train_line(slurm_tpl, str(ypath))
        s = re.sub(r"^#SBATCH --output=.*$", "#SBATCH --output={}".format(outs / (tag + ".out")), s, flags=re.MULTILINE)
        s = re.sub(r"^#SBATCH --error=.*$", "#SBATCH --error={}".format(outs / (tag + ".err")), s, flags=re.MULTILINE)
        if not s.startswith("#!/bin/bash"):
            s = "#!/bin/bash\n" + s
        spath.write_text(s, encoding="utf-8")

        ok, submit_output, job_id = False, "not submitted", None
        if args.submit:
            ok, submit_output, job_id = submit_sbatch(spath)

        rows.append({
            "lr": lr_s,
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

