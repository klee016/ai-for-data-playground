#!/bin/bash

for pid in $(pgrep -af jupyter-wb575476 | grep ai_for_data_playground | awk '{print $1}'); do
    echo "Killing process with PID $pid"
    kill -9 $pid
done

echo "AI for Data Playground stopped."