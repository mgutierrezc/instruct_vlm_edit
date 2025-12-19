#!/bin/bash
cd "$(dirname "$0")"

sbatch qwen.sbatch
sbatch qwen_4b.sbatch
sbatch blip.sbatch
sbatch llava.sbatch

