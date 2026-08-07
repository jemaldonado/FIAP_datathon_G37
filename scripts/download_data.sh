#!/usr/bin/env bash
set -euo pipefail

# Load credentials from .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
  echo "ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env"
  exit 1
fi

mkdir -p data/kaggle/bank-marketing
mkdir -p data/kaggle/bank-marketing-janiobachmann
mkdir -p data/kaggle/bank-term-deposit-dharmik34
mkdir -p data/kaggle/telemarketing-aguado

echo "Downloading henriqueyamahata/bank-marketing (primary)..."
kaggle datasets download -d henriqueyamahata/bank-marketing \
  -p data/kaggle/bank-marketing --unzip

echo "Downloading janiobachmann/bank-marketing-dataset..."
kaggle datasets download -d janiobachmann/bank-marketing-dataset \
  -p data/kaggle/bank-marketing-janiobachmann --unzip

echo "Downloading dharmik34/bank-term-deposit-subscription..."
kaggle datasets download -d dharmik34/bank-term-deposit-subscription \
  -p data/kaggle/bank-term-deposit-dharmik34 --unzip

echo "Downloading aguado/telemarketing-jyb-dataset..."
kaggle datasets download -d aguado/telemarketing-jyb-dataset \
  -p data/kaggle/telemarketing-aguado --unzip

echo "Done. Raw data in data/kaggle/"
