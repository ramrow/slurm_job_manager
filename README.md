# SLURM Job Manager (LR Sweep + W&B Summary)

This folder contains two helper scripts:

- `generate_lr_jobs.py` — creates YAML + SLURM files for LR sweep, optional `sbatch` submit
- `summarize_wandb_runs.py` — summarizes W&B runs for the generated jobs

## Files

- `config.yaml` (template YAML)
- `run_example.slurm.sh` (template SLURM)
- `generate_lr_jobs.py`
- `summarize_wandb_runs.py`

---

## What `generate_lr_jobs.py` does

Creates 3 directories:

- `yamls/` — generated YAML configs
- `slurms/` — generated SLURM scripts
- `slurm_outputs/` — stdout/stderr output paths for each job

Generates 10 learning rates:

- `5e-7`, `1e-6`, `2e-6`, `3e-6`, `5e-6`, `7e-6`, `1e-5`, `1.5e-5`, `2e-5`, `3e-5`

For each LR:

- creates matching YAML + SLURM files
- updates `learning_rate` in YAML
- updates `output_dir` in YAML to keep runs separate
- points SLURM output/error to `slurm_outputs/`
- optionally submits with `sbatch`

Writes a manifest file:

- `manifest_YYYYMMDD_HHMMSS.json`

This maps each LR to its YAML, SLURM, output files, and job ID (if submitted).

---

## How to run

### 1) Generate files only

```bash
python generate_lr_jobs.py
```

### 2) Generate and submit jobs

```bash
python generate_lr_jobs.py --submit
```

---

## W&B summary script

After jobs have started/finished, run:

```bash
python summarize_wandb_runs.py --entity <YOUR_WANDB_ENTITY> --project <YOUR_WANDB_PROJECT> --manifest manifest_YYYYMMDD_HHMMSS.json
```

It will:

- read the manifest
- query W&B runs
- match runs by `learning_rate`
- write `wandb_lr_summary.json`

---

## Quick tips

- List manifests:
  - Linux/macOS: `ls manifest_*.json`
  - PowerShell: `dir manifest_*.json`

- If `sbatch` is unavailable locally, run with `--submit` on your cluster/login node.

- If W&B summary is empty, confirm:
  - your run configs include `learning_rate`
  - entity/project are correct
  - runs have synced to W&B
