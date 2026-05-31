#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
  echo "Missing .env file. Copy .env.example to .env and fill it in."
  exit 1
fi

set -a
source .env
set +a

echo "Checking bucket:"
echo "  bucket:   $S3_BUCKET"
echo "  endpoint: $S3_ENDPOINT_URL"

s5cmd --endpoint-url "$S3_ENDPOINT_URL" ls "s3://$S3_BUCKET/"

echo "Storage check passed."
