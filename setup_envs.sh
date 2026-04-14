#!/usr/bin/env bash
# =============================================================================
# setup_envs.sh
# Creates one Conda environment per Qiskit version and installs dependencies.
#
# Usage:
#   chmod +x setup_envs.sh
#   ./setup_envs.sh
#
# After this runs you should have three environments:
#   qiskit-v0  (Qiskit 0.43.3)
#   qiskit-v1  (Qiskit 1.3.2)
#   qiskit-v2  (Qiskit 2.0.0)
# =============================================================================

set -e  # Exit immediately on any error

PYTHON_VERSION="3.10"  # 3.10 is compatible with all three Qiskit versions

COMMON_PACKAGES=(
    "pylatexenc"
    "matplotlib"
)

echo "======================================================"
echo " Quantum API Drift -- Conda Environment Setup"
echo "======================================================"

# Helper: create the env if needed, then ensure the requested packages exist.
create_env() {
    local ENV_NAME=$1
    shift
    local PACKAGES=("$@")

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "[SYNC] Environment '${ENV_NAME}' already exists."
    else
        echo ""
        echo "[CREATE] ${ENV_NAME} ..."
        conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    fi

    echo "[INSTALL] ${ENV_NAME}: ${PACKAGES[*]}"
    conda run -n "${ENV_NAME}" python -m pip install "${PACKAGES[@]}"
    echo "[OK] ${ENV_NAME} ready."
}

# Qiskit 0.43.x still bundles Aer through the meta-package.
create_env "qiskit-v0" \
    "qiskit==0.43.3" \
    "${COMMON_PACKAGES[@]}"

# Qiskit 1.x and 2.x require Aer to be installed separately for local
# simulator/statevector/noise-model problems in raw_problems.json.
create_env "qiskit-v1" \
    "qiskit==1.3.2" \
    "qiskit-aer" \
    "${COMMON_PACKAGES[@]}"

create_env "qiskit-v2" \
    "qiskit==2.0.0" \
    "qiskit-aer" \
    "${COMMON_PACKAGES[@]}"

echo ""
echo "======================================================"
echo " All environments ready. Verify with:"
echo "   conda env list"
echo "   conda run -n qiskit-v0 python -c 'import qiskit; print(qiskit.__qiskit_version__)'"
echo "   conda run -n qiskit-v1 python -c 'import qiskit; print(qiskit.__version__)'"
echo "   conda run -n qiskit-v2 python -c 'import qiskit; print(qiskit.__version__)'"
echo "======================================================"
