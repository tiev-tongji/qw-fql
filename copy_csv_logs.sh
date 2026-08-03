#!/bin/bash
# 将 logs 下所有 train.csv / eval.csv 按原始相对目录结构复制到 logs_test
set -euo pipefail

SRC_DIR="logs"
DST_DIR="logs_test"

if [ ! -d "$SRC_DIR" ]; then
    echo "错误: 源目录 '$SRC_DIR' 不存在" >&2
    exit 1
fi

count=0
while IFS= read -r -d '' f; do
    rel="${f#$SRC_DIR/}"
    dst="$DST_DIR/$rel"
    mkdir -p "$(dirname "$dst")"
    cp -f "$f" "$dst"
    count=$((count + 1))
done < <(find "$SRC_DIR" -type f \( -name "train.csv" -o -name "eval.csv" \) -print0)

echo "共复制 $count 个文件到 '$DST_DIR'"
