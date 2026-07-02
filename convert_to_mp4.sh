#!/usr/bin/env bash
# google slides didn't like the AVI's codec, so converting

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_video> <output_video>"
    exit 1
fi

input_path="$1"
output_path="$2"

ffmpeg -i "$input_path" \
    -c:v libx264 \
    -pix_fmt yuv420p \
    "$output_path"