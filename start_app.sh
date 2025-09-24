#!/bin/bash
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate ai_for_data

# export GRADIO_TEMP_DIR="tmp"

LOG_DIR="log"

if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

CURRENT_DATE=$(date +%F)
LOG_FILE="${LOG_DIR}/${CURRENT_DATE}.log"

uvicorn ai_for_data_playground_main:app --port 8082 --host 0.0.0.0 > "$LOG_FILE" 2>&1 &
# uvicorn ai_for_data_playground_main:app --port 8082 --host 0.0.0.0
# uvicorn ai_for_data_playground_main:app --port 8082 --host 0.0.0.0 --ssl-keyfile=/etc/ssl/private/w1lxscirender02.worldbank.org.key --ssl-certfile=/etc/ssl/certs/w1lxscirender02.worldbank.org.pem
echo "AI for Data Playground started, logging to $LOG_FILE"