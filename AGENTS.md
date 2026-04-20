# Repository Guidelines

## Skills
A skill is a reusable local workflow stored in a `SKILL.md` file.

### Available skills
- `post-test-catch`: Finalize a completed and fully tested change by committing the intended files, archiving the latest commit into `tmp/catch_<timestamp>/`, and reporting the commit SHA plus archive path. Use when the user says `走 catch 流程`, `catch 一下`, `提交并归档`, `收口提交`, or asks for post-test wrap-up. (file: `D:/SAR_Nir_dt/StreamYOLO/.codex/skills/post-test-catch/SKILL.md`)

### How to use skills
- If the user names the skill or the request clearly matches the described workflow, open the referenced `SKILL.md` and follow it.
- Keep the skill scoped to this repository only; do not assume it applies to sibling projects unless the user asks for that project too.
- Read only the files referenced by the skill that are needed for the current task.

## Project Structure & Module Organization
`cfgs/` contains the VisDrone training configs. Dataset definitions and evaluators live under `exps/`, especially `exps/data/`, `exps/dataset/`, and `exps/evaluators/`. Entrypoints such as training and evaluation live in `tools/`, while reusable shell launchers live in `scripts/`. Generated training outputs go to `outputs/`, pretrained weights live in `pretrained/`, and temporary local artifacts belong under `tmp/`.

## Build, Test, and Development Commands
Create or activate the validated environment from the repo root:
```powershell
conda create -p .\.conda\streamyolo_visdrone_py310_cu128 python=3.10.18 pip=25.3 setuptools=80.9.0 wheel=0.45.1 -y
conda activate D:\SAR_Nir_dt\StreamYOLO\.conda\streamyolo_visdrone_py310_cu128
pip install -r requirements.txt
pip install --no-deps https://github.com/Megvii-BaseDetection/YOLOX/archive/refs/tags/0.3.0.zip
```
Set `PYTHONPATH` to the repo root before local runs. Train with `python tools/train.py -f cfgs/visdrone_m_s50_onex_dfp_tal_flip.py -d 1 -b 8 -c <checkpoint> --fp16` and evaluate with `python tools/eval.py -f cfgs/visdrone_m_s50_onex_dfp_tal_flip.py -c <checkpoint> -d 1 -b 8 --conf 0.01 --fp16`. Use `scripts/train_visdrone.sh` for WSL or Linux runs.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and lowercase module names. Keep new config files aligned with the existing `cfgs/visdrone_*` naming pattern. Match the surrounding import grouping and avoid broad refactors unless the task requires them.

## Testing Guidelines
This repo does not ship a dedicated unit-test suite, so validate changes with the narrowest realistic training, evaluation, or dataset-construction command for the code path you touched. For config or data-pipeline edits, prefer a small smoke run over a full training cycle when possible. Report the exact command, dataset root, checkpoint, and device count used for validation.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects focused on one change. In summaries or PR notes, describe the problem, the specific config or code path changed, and the exact validation command you ran. Do not commit generated outputs under `outputs/` or temporary artifacts under `tmp/`.
