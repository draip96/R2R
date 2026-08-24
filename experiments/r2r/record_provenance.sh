#!/bin/bash
set -euo pipefail

R2R_ROOT=${R2R_ROOT:-/project/6101829/draip/R2R}
PROVENANCE_DIR=${1:?provenance directory is required}
mkdir -p "${PROVENANCE_DIR}"
cd "${R2R_ROOT}"
git rev-parse HEAD > "${PROVENANCE_DIR}/commit.txt"
git status --porcelain --untracked-files=all > "${PROVENANCE_DIR}/git-status.txt"
git diff --binary HEAD -- > "${PROVENANCE_DIR}/working-tree.patch"
module -t list > "${PROVENANCE_DIR}/modules.txt" 2>&1 || true
nvidia-smi -q > "${PROVENANCE_DIR}/nvidia-smi.txt"
sha256sum \
  recall2imagine/agent.py \
  recall2imagine/embodied/replay/generic_lfs.py \
  recall2imagine/embodied/replay/state_adjoint_cache.py \
  recall2imagine/ssm/common.py \
  > "${PROVENANCE_DIR}/source-sha256.txt"
