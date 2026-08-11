#!/usr/bin/env bash
# Download the Bank Marketing dataset into data/kaggle/ (Linux, macOS, Git Bash, WSL).
#
# Usage:
#   scripts/download_data.sh           # skip the download if the CSV is already there
#   scripts/download_data.sh --force   # re-download even when the CSV is present
#
# The cross-platform equivalent, which also builds the parquet layer, is:
#   python scripts/download_data.py
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FORCE=0
if [ "${1:-}" = "--force" ]; then
  FORCE=1
fi

# Load credentials from .env if present
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
  if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
    cat >&2 <<'EOF'
ERROR: Kaggle credentials not found.

1. Sign in at https://www.kaggle.com and open Settings > API > Create New Token.
2. Copy .env.example to .env in the repository root.
3. Put the username and key from the downloaded kaggle.json in .env:

       KAGGLE_USERNAME=your_kaggle_username
       KAGGLE_KEY=your_kaggle_api_key

4. Run this script again.

A KAGGLE_API_TOKEN access token in .env is accepted as an alternative to the pair above.
Never commit .env - it is listed in .gitignore.
EOF
    exit 1
  fi
fi

SLUG="henriqueyamahata/bank-marketing"
TARGET="data/kaggle/bank-marketing"

mkdir -p "$TARGET"

if [ "$FORCE" -eq 0 ] && compgen -G "$TARGET/*.csv" > /dev/null; then
  echo "Skipping $SLUG - CSV already in $TARGET"
else
  echo "Downloading $SLUG -> $TARGET"
  if ! kaggle datasets download -d "$SLUG" -p "$TARGET" --unzip; then
    echo "ERROR: download of $SLUG failed" >&2
    echo "  Manual alternative: https://www.kaggle.com/datasets/$SLUG -> extract into $TARGET" >&2
    exit 1
  fi
fi

echo "Done. Raw data in data/kaggle/"
echo "Now build the processed layer: python scripts/download_data.py"
