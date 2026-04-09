#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_SIZE="${MODEL_SIZE:-m}"
DATA_ROOT="${STREAMYOLO_VISDRONE_ROOT:-/mnt/e/VOD-dataset/VisDrone_MOT_TransVOD}"
DEVICES="${DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_EPOCH="${MAX_EPOCH:-15}"
FP16="${FP16:-1}"
OCCUPY="${OCCUPY:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_ROOT}/pretrained}"
CKPT="${CKPT:-}"

export STREAMYOLO_VISDRONE_ROOT="${DATA_ROOT}"

case "${MODEL_SIZE}" in
  s)
    EXP_FILE="${REPO_ROOT}/cfgs/visdrone_s_s50_onex_dfp_tal_flip.py"
    DEFAULT_CKPT="${PRETRAINED_DIR}/yolox_s.pth"
    ;;
  m)
    EXP_FILE="${REPO_ROOT}/cfgs/visdrone_m_s50_onex_dfp_tal_flip.py"
    DEFAULT_CKPT="${PRETRAINED_DIR}/yolox_m.pth"
    ;;
  l)
    EXP_FILE="${REPO_ROOT}/cfgs/visdrone_l_s50_onex_dfp_tal_flip.py"
    DEFAULT_CKPT="${PRETRAINED_DIR}/yolox_l.pth"
    ;;
  *)
    echo "Unsupported MODEL_SIZE=${MODEL_SIZE}. Use s, m, or l." >&2
    exit 1
    ;;
esac

if [[ -z "${CKPT}" && -f "${DEFAULT_CKPT}" ]]; then
  CKPT="${DEFAULT_CKPT}"
fi

CMD=(
  python
  "${REPO_ROOT}/tools/train.py"
  -f "${EXP_FILE}"
  -d "${DEVICES}"
  -b "${BATCH_SIZE}"
)

if [[ -n "${CKPT}" ]]; then
  CMD+=(-c "${CKPT}")
fi

if [[ "${FP16}" == "1" ]]; then
  CMD+=(--fp16)
fi

if [[ "${OCCUPY}" == "1" ]]; then
  CMD+=(-o)
fi

if [[ -n "${EXPERIMENT_NAME}" ]]; then
  CMD+=(-expn "${EXPERIMENT_NAME}")
fi

CMD+=(
  data_num_workers "${NUM_WORKERS}"
  max_epoch "${MAX_EPOCH}"
)

if (( "$#" > 0 )); then
  CMD+=("$@")
fi

echo "Repo root: ${REPO_ROOT}"
echo "Data root: ${STREAMYOLO_VISDRONE_ROOT}"
echo "Model size: ${MODEL_SIZE}"
echo "Checkpoint: ${CKPT:-<none>}"
printf 'Run command:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
