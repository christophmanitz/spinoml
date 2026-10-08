#!/usr/bin/env bash
# SpinoML — one-command dev environment for a fresh machine.
#
#   git clone https://github.com/christophmanitz/spinoml.git && cd spinoml
#   bash scripts/setup-dev.sh            # CPU torch, env "spinoml-dev"
#   conda activate spinoml-dev && npm run ci
#
# What it builds (mirrors the dev box and .github/workflows/ci.yml):
#   * conda env (conda-forge only): python 3.12, nodejs 20, rust 1.96
#   * torch 2.12.0 (CPU build by default) + sidecar-torch/requirements.txt
#   * npm ci for the root project and for sidecar-llm/
#   * checks the Tauri system libraries (apt packages) and reports what is missing
#
# What it does NOT do (needs you, see the final checklist): CLI logins (claude,
# opencode), ~/.ssh/config aliases, API keys (they live in the app's
# localStorage), cluster accounts, a GPU driver.
#
# Idempotent: re-running skips what is already there. Nothing runs under sudo
# unless you pass --apt.
set -euo pipefail

ENV_NAME="${SPINOML_CONDA_ENV:-spinoml-dev}"
PY_SPEC="${SPINOML_PY_SPEC:-python=3.12}"
NODE_SPEC="${SPINOML_NODE_SPEC:-nodejs=20}"
RUST_SPEC="${SPINOML_RUST_SPEC:-rust=1.96}"
TORCH_VERSION="${SPINOML_TORCH_VERSION:-2.12.0}"
TORCH_INDEX="${SPINOML_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

DRY_RUN=0 DO_APT=0 RECREATE=0 DO_CARGO=0 DO_SMOKE=0

APT_PACKAGES=(
  build-essential pkg-config patchelf libssl-dev
  libwebkit2gtk-4.1-dev libjavascriptcoregtk-4.1-dev libsoup-3.0-dev
  libgtk-3-dev librsvg2-dev libayatana-appindicator3-dev
)
# pkg-config module names that the apt packages above provide.
PKGCONFIG_MODULES=(
  webkit2gtk-4.1 javascriptcoregtk-4.1 libsoup-3.0 gtk+-3.0 librsvg-2.0
  ayatana-appindicator3-0.1 openssl
)

usage() {
  cat <<EOF
Usage: bash scripts/setup-dev.sh [options]

  --env NAME             conda env name (default: $ENV_NAME, or \$SPINOML_CONDA_ENV)
  --cuda [cu130]         install the CUDA torch build instead of CPU (default tag cu130)
  --torch-index URL      explicit pip index for torch (default: the CPU index)
  --torch-version V      torch version (default: $TORCH_VERSION)
  --apt                  install the missing Tauri system libraries with sudo apt-get
  --recreate             delete and rebuild the conda env
  --cargo                also run 'cargo check' in src-tauri (long first build, ~GBs)
  --smoke                afterwards run: npm run build, verify:codegen, verify:sidecar
  --dry-run              print every command, change nothing
  -h, --help             this text

Environment overrides: SPINOML_CONDA_ENV, SPINOML_PY_SPEC, SPINOML_NODE_SPEC,
SPINOML_RUST_SPEC, SPINOML_TORCH_VERSION, SPINOML_TORCH_INDEX.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --env) ENV_NAME="${2:?--env needs a name}"; shift 2 ;;
    --cuda)
      tag="cu130"
      if [ $# -gt 1 ] && [[ "${2}" == cu* ]]; then tag="$2"; shift; fi
      TORCH_INDEX="https://download.pytorch.org/whl/$tag"; shift ;;
    --torch-index) TORCH_INDEX="${2:?--torch-index needs a URL}"; shift 2 ;;
    --torch-version) TORCH_VERSION="${2:?--torch-version needs a version}"; shift 2 ;;
    --apt) DO_APT=1; shift ;;
    --recreate) RECREATE=1; shift ;;
    --cargo) DO_CARGO=1; shift ;;
    --smoke) DO_SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -t 1 ]; then B=$'\033[1m' G=$'\033[32m' Y=$'\033[33m' R=$'\033[31m' N=$'\033[0m'; else B= G= Y= R= N=; fi
step() { printf '\n%s== %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '  %s✗ %s%s\n' "$R" "$*" "$N" >&2; exit 1; }

# run CMD…: echo it, execute unless --dry-run.
run() {
  printf '  $ %s\n' "$*"
  if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -f package.json ] && [ -d sidecar-torch ] || die "not a SpinoML checkout: $REPO"

# ── 1. conda ────────────────────────────────────────────────────────────────
step "conda"
CONDA="$(command -v conda || true)"
if [ -z "$CONDA" ]; then
  for c in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
           "$HOME/mambaforge/bin/conda" /opt/conda/bin/conda; do
    if [ -x "$c" ]; then CONDA="$c"; break; fi
  done
fi
[ -n "$CONDA" ] || die "conda not found. Install Miniforge first:
    https://github.com/conda-forge/miniforge#install   (then re-run this script)"
ok "conda: $CONDA ($("$CONDA" --version))"

# ── 2. system libraries (Tauri) ─────────────────────────────────────────────
step "system libraries for the Tauri shell"
missing_tools=()
for t in git cc; do command -v "$t" >/dev/null 2>&1 || missing_tools+=("$t"); done
missing_mods=()
if command -v pkg-config >/dev/null 2>&1; then
  for m in "${PKGCONFIG_MODULES[@]}"; do pkg-config --exists "$m" 2>/dev/null || missing_mods+=("$m"); done
else
  missing_tools+=(pkg-config)
fi
if [ ${#missing_tools[@]} -eq 0 ] && [ ${#missing_mods[@]} -eq 0 ]; then
  ok "all Tauri build libraries present"
else
  [ ${#missing_tools[@]} -eq 0 ] || warn "missing tools: ${missing_tools[*]}"
  [ ${#missing_mods[@]} -eq 0 ] || warn "missing pkg-config modules: ${missing_mods[*]}"
  if command -v apt-get >/dev/null 2>&1; then
    if [ "$DO_APT" -eq 1 ]; then
      run sudo apt-get update
      run sudo apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
    else
      warn "install them with:  sudo apt-get install -y ${APT_PACKAGES[*]}"
      warn "(or re-run with --apt). Only the Rust/Tauri build needs them — the web UI, sidecars and"
      warn "most test suites work without."
    fi
  else
    warn "no apt-get here — see https://v2.tauri.app/start/prerequisites/#linux for your distro"
  fi
fi

# ── 3. conda env ────────────────────────────────────────────────────────────
step "conda env '$ENV_NAME'"
env_exists() { "$CONDA" env list | awk -v n="$ENV_NAME" '$1 == n { f = 1 } END { exit !f }'; }
if env_exists && [ "$RECREATE" -eq 1 ]; then
  run "$CONDA" env remove -y -n "$ENV_NAME"
fi
if env_exists && [ "$RECREATE" -eq 0 ]; then
  ok "env exists — keeping it (use --recreate to rebuild)"
else
  run "$CONDA" create -y -n "$ENV_NAME" --override-channels -c conda-forge "$PY_SPEC" "$NODE_SPEC" "$RUST_SPEC"
fi

if [ "$DRY_RUN" -eq 1 ] && ! env_exists; then
  PREFIX="<prefix of $ENV_NAME>"
else
  PREFIX="$("$CONDA" run --no-capture-output -n "$ENV_NAME" sh -c 'printf %s "$CONDA_PREFIX"')"
  [ -x "$PREFIX/bin/python" ] || die "env '$ENV_NAME' has no python at $PREFIX/bin"
  export PATH="$PREFIX/bin:$PATH"
  # ~/.local/lib/pythonX.Y/site-packages is visible in EVERY env of that Python version; without this pip
  # treats a user-site numpy/requests as "already installed" and the env silently depends on it.
  export PYTHONNOUSERSITE=1
  unset PIP_REQUIRE_VIRTUALENV PYTHONPATH
  ok "python $(python --version 2>&1 | cut -d' ' -f2), node $(node --version), cargo $(cargo --version | cut -d' ' -f2)"
fi

# ── 4. python deps ──────────────────────────────────────────────────────────
step "python deps (torch $TORCH_VERSION from $TORCH_INDEX)"
run python -m pip install --upgrade pip
run python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
run python -m pip install -r sidecar-torch/requirements.txt

# ── 5. node deps ────────────────────────────────────────────────────────────
step "node deps"
run npm ci
run npm ci --prefix sidecar-llm

# ── 6. verify the install ───────────────────────────────────────────────────
step "verify"
if [ "$DRY_RUN" -eq 1 ]; then
  warn "dry run — nothing installed, nothing verified"
else
  python - <<'PY' || die "python imports failed (see above)"
import numpy, pandas, rdkit, sklearn, torch, torch_geometric
cuda = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if cuda else "none"
print(f"  ✓ torch {torch.__version__} (CUDA available: {cuda}, device: {dev}), "
      f"pyg {torch_geometric.__version__}, rdkit {rdkit.__version__}, numpy {numpy.__version__}")
PY
  [ -d node_modules/.bin ] && [ -d sidecar-llm/node_modules ] && ok "node_modules present (root + sidecar-llm)" \
    || die "node_modules missing after npm ci"
fi

if [ "$DO_CARGO" -eq 1 ]; then
  step "cargo check (src-tauri)"
  run bash -c 'cd src-tauri && cargo check'
fi

if [ "$DO_SMOKE" -eq 1 ]; then
  step "smoke: build + codegen + sidecar"
  run npm run build
  run npm run verify:codegen
  run npm run verify:sidecar
fi

# ── 7. what is still on you ─────────────────────────────────────────────────
step "optional tools on this machine"
for t in ssh tmux opencode claude sbatch; do
  if command -v "$t" >/dev/null 2>&1; then ok "$t"; else warn "$t not found"; fi
done

cat <<EOF

${B}Done.${N} Next:

  conda activate $ENV_NAME
  npm run ci                  # full suite pyramid (add --allow-known on a box without cluster/GPU)
  npm run tauri dev           # native app (needs the system libraries above)

Not set up by this script — do these by hand if you need them:
  [ ] claude login / opencode login      (LLM suites: test:llm-live, verify:opencode)
  [ ] ~/.ssh/config alias for the cluster (test:remote-live; key-based, no password prompt)
  [ ] NVIDIA driver + --cuda re-run        (test:hardware-cuda; otherwise it prints SKIPPED CUDA)
  [ ] API keys / provider choice           (stored per machine in the app's localStorage)
  [ ] Chromium for test:webview-csp        (SKIPPED without a browser; or set SPINOML_CHROMIUM)
EOF
if [ "$ENV_NAME" != "mlforge-dev" ]; then
  printf '\nNote: a few older docs/suites still say "mlforge-dev". The runner finds your env when it is\nactivated, or via  SPINOML_CONDA_ENV=%s npm run ci\n' "$ENV_NAME"
fi
