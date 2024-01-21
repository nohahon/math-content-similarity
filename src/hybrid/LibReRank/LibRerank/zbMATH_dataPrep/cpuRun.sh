#!/bin/sh
#SBATCH --job-name=genDataSlow
#SBATCH --partition=normal
#SBATCH --time=0-24:5:0   # days-hours:minutes:seconds
#SBATCH --nodes=8       # at least 4 nodes, up to 8
#SBATCH --ntasks=16       # 16 processes
#SBATCH --mem-per-cpu=3840 # in MB, up to 4GB per core
#srun hostname | sort

source ~/.bashrc
conda activate myenvLLM

python embeDDings_gen.py
#python run_reranker.py --setting_path=./example/config/ad/prm_setting.json
#python run_init_ranker.py --setting_path=./example/config/ad/mart_setting.json
