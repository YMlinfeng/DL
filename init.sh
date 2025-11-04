#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
conda activate quick_start
echo "🔍 当前Python路径: $(which python)"
echo "🐍 Python版本: $(python --version 2>&1)"

git config --global user.name "YMlinfeng"
git config --global user.email "xiao102851@163.com"
