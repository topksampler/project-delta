#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .env ]; then
  echo "Missing .env file. Copy .env.example to .env and fill it in."
  exit 1
fi

set -a
source .env
set +a

TEST_ID="$(hostname)-$(date +%Y%m%d-%H%M%S)"
LOCAL_DIR="tmp/storage-test-$TEST_ID"
REMOTE_PATH="s3://$S3_BUCKET/tests/$TEST_ID/hello.txt"

mkdir -p "$LOCAL_DIR"
echo "hello from $TEST_ID" > "$LOCAL_DIR/hello.txt"

echo "Uploading test file..."
s5cmd --endpoint-url "$S3_ENDPOINT_URL" cp "$LOCAL_DIR/hello.txt" "$REMOTE_PATH"

echo "Listing remote test path..."
s5cmd --endpoint-url "$S3_ENDPOINT_URL" ls "s3://$S3_BUCKET/tests/$TEST_ID/"

echo "Downloading test file..."
s5cmd --endpoint-url "$S3_ENDPOINT_URL" cp "$REMOTE_PATH" "$LOCAL_DIR/downloaded.txt"

echo "Downloaded content:"
cat "$LOCAL_DIR/downloaded.txt"

echo "Deleting remote test file..."
s5cmd --endpoint-url "$S3_ENDPOINT_URL" rm "$REMOTE_PATH"

echo "Roundtrip test passed."
