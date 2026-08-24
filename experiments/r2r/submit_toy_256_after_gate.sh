#!/bin/bash
#SBATCH --account=aip-valenzan
#SBATCH --partition=cpubase_b1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:10
#SBATCH --job-name=r2r-toy-stage
#SBATCH --output=experiments/r2r/toy-stage-%j.out

set -euo pipefail
R2R_ROOT=/project/6101829/draip/R2R
RESULT_ROOT=${R2R_RESULT_ROOT:-${R2R_ROOT}/experiments/r2r/results}
CAMPAIGN=${R2R_CAMPAIGN:?}
FIRST=${RESULT_ROOT}/toy/${CAMPAIGN}/horizon128/SUCCESS
if [[ ! -f "${FIRST}" ]]; then
  echo "128-transition ToyMemory gate is not successful" >&2
  exit 4
fi
cd "${R2R_ROOT}"
second=$(sbatch --parsable \
  --export=ALL,R2R_CAMPAIGN="${CAMPAIGN}",R2R_TOY_HORIZON=256 \
  experiments/r2r/slurm_toy.sh)
printf '%s\n' "${second}" > \
  "${RESULT_ROOT}/toy/${CAMPAIGN}/job256.txt"
printf '%s\n' "${second}"
