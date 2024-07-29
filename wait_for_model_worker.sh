#!/bin/bash

LOG_FILE="/app/logs/model_worker"
MAX_ATTEMPTS=60
SLEEP_TIME=3

echo "Waiting for model worker to be ready..."

for ((i = 1; i <= MAX_ATTEMPTS; i++)); do
  if grep -q "Send heart beat. Models:" "$LOG_FILE"*; then
    echo "Model worker is ready!"
    exec "$@"
  fi
  echo "Attempt $i: Model worker not ready yet. Waiting $SLEEP_TIME seconds..."
  sleep $SLEEP_TIME
done

echo "Model worker did not become ready in time. Starting web interface anyway..."
exec "$@"
