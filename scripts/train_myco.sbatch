#!/bin/bash
#SBATCH --job-name=myco_v3_ssl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=PATHgpu,gpu-medium,gpu-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

###############################################################################
# MyCo v3 SSL training (MoCo v3) - Slurm sbatch script
#
# Required inputs (format):
#   1) WSI_DIR: directory containing WSI files (e.g., .svs, .tif)
#      Example:
#        /data/wsis/slide_001.svs
#        /data/wsis/slide_002.svs
#   2) ANN_DIR: directory containing annotations per slide (.xml or .geojson)
#      Filenames must match slide IDs from WSI_DIR (e.g., slide_001.xml)
#   3) SLIDE_LABELS: CSV file with columns: slide_id,label
#      Labels can be BID/MF or 0/1. Example:
#        slide_id,label
#        slide_001,BID
#        slide_002,MF
#   4) OUTDIR: output directory for checkpoints, logs, mosaics, and metrics
###############################################################################


export WSI_DIR="/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/WSI_database_cropped"
export ANN_DIR="/exports/path-cutane-lymfomen-hpc/siemen/MIRAGE/MIRAGE/outputs/segmentations"
export SLIDE_LABELS="/exports/path-cutane-lymfomen-hpc/Thom_Doeleman/WSI_database_cropped/annotations_MF_BID_dec2025_T1.csv"
export OUTDIR="./outputs/myco_v3_ssl"

WSI_DIR="${WSI_DIR:?Set WSI_DIR to your WSI directory}"
ANN_DIR="${ANN_DIR:?Set ANN_DIR to your annotation directory}"
SLIDE_LABELS="${SLIDE_LABELS:?Set SLIDE_LABELS to labels CSV}"
OUTDIR="${OUTDIR:?Set OUTDIR to an output directory}"

mkdir -p "${OUTDIR}"

# Optional: load modules or activate your environment here.
# module load cuda/12.1
# source /path/to/venv/bin/activate

echo "Starting MyCo v3 SSL training at $(date)"
echo "WSI_DIR=${WSI_DIR}"
echo "ANN_DIR=${ANN_DIR}"
echo "SLIDE_LABELS=${SLIDE_LABELS}"
echo "OUTDIR=${OUTDIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

export PYTHONUNBUFFERED=1

# Log progress/metrics to stdout+file, and CSV metrics to OUTDIR/logs.
LOG_FILE="${OUTDIR}/train_${SLURM_JOB_ID:-manual}.log"

srun \
  uv run ../train_model.py \
    --wsi_dir "${WSI_DIR}" \
    --ann_dir "${ANN_DIR}" \
    --slide_labels "${SLIDE_LABELS}" \
    --outdir "${OUTDIR}" \
    --devices "${SLURM_GPUS_ON_NODE:-1}" \
  2>&1 | tee "${LOG_FILE}"

echo "Finished at $(date)"
