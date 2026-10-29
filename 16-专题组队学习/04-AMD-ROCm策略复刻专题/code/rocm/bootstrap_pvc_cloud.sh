#!/usr/bin/env bash
set -Eeuo pipefail

# One-time bootstrap for a fresh Radeon Cloud PVC notebook.
# Run this from the JupyterLab Terminal as root or as a sudo-capable user.

ROOT_DIR="/workspace"
REPO_DIR="$ROOT_DIR/every-embodied"
ENV_DIR="$ROOT_DIR/envs/lerobot-rocm"
CACHE_DIR="$ROOT_DIR/cache"

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

echo "[1/6] Preparing persistent directories"
mkdir -p "$ROOT_DIR" "$ROOT_DIR"/{envs,cache,artifacts,runs,logs}
df -h "$ROOT_DIR"

echo "[2/6] Installing and starting SSH service"
if ! command -v sshd >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || [ ! -s /etc/ssl/certs/ca-certificates.crt ]; then
  as_root apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server git curl ca-certificates rsync tmux iproute2
fi
as_root mkdir -p /run/sshd
if ! pgrep -x sshd >/dev/null 2>&1; then
  as_root /usr/sbin/sshd
fi
sshd -T 2>/dev/null | grep -E '^(port|pubkeyauthentication|permitrootlogin) ' || true

echo "[3/6] Checking AMD ROCm"
command -v rocminfo >/dev/null 2>&1 && rocminfo | grep -m 3 -E 'Name:|Marketing Name:' || echo "rocminfo unavailable"
if command -v amd-smi >/dev/null 2>&1; then
  amd-smi static --gpu 0 2>/dev/null || amd-smi 2>/dev/null || true
elif command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi 2>/dev/null || true
else
  echo "amd-smi/rocm-smi unavailable"
fi

echo "[4/6] Preparing repository"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/datawhalechina/every-embodied.git "$REPO_DIR"
  git -C "$REPO_DIR" sparse-checkout init --cone
  git -C "$REPO_DIR" sparse-checkout set '16-专题组队学习/04-AMD-ROCm策略复刻专题'
  git -C "$REPO_DIR" checkout main
else
  echo "Repository already exists on PVC; preserving local tutorial changes."
fi

echo "[5/6] Preparing a persistent Python environment"
if ! command -v python3 >/dev/null 2>&1; then
  as_root apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
  as_root apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip iproute2
fi
python3 -m venv --system-site-packages "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
mkdir -p "$CACHE_DIR"/{pip,huggingface,torch_extensions}
cat > "$ROOT_DIR/rocm_pvc_env.sh" <<EOF
export REPO_DIR="$REPO_DIR"
export VENV="$ENV_DIR"
export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$CACHE_DIR/huggingface/hub"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export PATH="$ENV_DIR/bin:\$PATH"
EOF

echo "[6/6] Writing persistent status"
cat > "$ROOT_DIR/PVC_STATUS.md" <<EOF
# Radeon Cloud PVC status

- Initialized: $(date -Is)
- Host: $(hostname)
- Workspace: $ROOT_DIR
- Repository: $REPO_DIR
- Environment: $ENV_DIR
- SSH: started by bootstrap script; verify with: ss -ltnp | grep ':22'

Next steps:

1. Source $ROOT_DIR/rocm_pvc_env.sh.
2. Run the ROCm/LeRobot verification notebook.
3. Sync the latest local tutorial files from the workstation.
4. Only then install model-specific dependencies and start training.
EOF

echo "DONE"
echo "Source: $ROOT_DIR/rocm_pvc_env.sh"
echo "Repository: $REPO_DIR"
echo "SSH check: ss -ltnp | grep ':22'"
