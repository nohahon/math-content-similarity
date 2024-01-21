#!/bin/sh
#SBATCH --job-name=genData
#SBATCH --partition=gpu
#SBATCH --account=etechnik_gpu
#SBATCH -N 1
#SBATCH --ntasks 2
#SBATCH --cpus-per-task 8
#SBATCH --gpus-per-task 1
#SBATCH --time=0-24:5:0   # days-hours:minutes:seconds

source ~/.bashrc
conda activate libRank

python embeDDings_gen.py
#python run_reranker.py --setting_path=./example/config/ad/prm_setting.json
#python run_init_ranker.py --setting_path=./example/config/ad/mart_setting.json
