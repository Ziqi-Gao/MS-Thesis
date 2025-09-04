#!/bin/bash
#SBATCH --account=p32737 
#SBATCH --job-name=get_w_all                # 作业名
#SBATCH --output=/gpfs/projects/p32737/del6500_home/CD/logs/get_w_all_%A_%a.out   # 标准输出，%A 为主作业号，%a 为数组索引
#SBATCH --error=/gpfs/projects/p32737/del6500_home/CD/logs/get_w_all_%A_%a.err    # 标准错误
#SBATCH --partition=gengpu                   # 分区名
#SBATCH --gres=gpu:1                        # 申请 1 张 GPU
#SBATCH --cpus-per-task=8                    # 申请 8 个 CPU 核心
#SBATCH --mem=256G                           # 申请 512G 内存
#SBATCH --time=4:00:00                      # 最长运行时间 4 小时
# #SBATCH --array=0-17                        # 2 模型 × 9 数据集 = 32 个子任务

# module purge
# module load cuda/11.8
# module load anaconda3
# eval "$(conda shell.bash hook)"
# conda activate /gpfs/projects/p32737/del6500_home/.conda/envs/cdenv
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


# cd /gpfs/projects/p32737/del6500_home/CD

# echo "[$(date)] Job started on node $(hostname)"
# START_TS=$(date +%s)
# ### —— 在脚本里定义两组参数数组 —— ###
# MODELS=( "meta-llama/Llama-2-7b-chat-hf" )
# DATASETS=( StrategyQA coinflip cities common counterfact hateeval STSA IMDb sarcasm )
# #DATASETS=(Action Bird City-district Comedy Cycling-competition Fish Football-match Horror Insect Island Joyful-tweets Motor-race Science Fiction Tennis-tournament Town Village)
# # 计算有多少个数据集
# NUM_DS=${#DATASETS[@]}

# # 通过数组索引派生出当前子任务所用的模型和数据集
# ARRAY_ID=$SLURM_ARRAY_TASK_ID
# MODEL_IDX=$(( ARRAY_ID / NUM_DS ))
# DATASET_IDX=$(( ARRAY_ID % NUM_DS ))

# MODEL="${MODELS[$MODEL_IDX]}"
# DATASET="${DATASETS[$DATASET_IDX]}"

# echo "[$SLURM_JOB_ID][$SLURM_ARRAY_TASK_ID] Running model=$MODEL on dataset=$DATASET"

# # —— 最终调用你的 main.py ——  #--concept "$DATASET" \#
# /gpfs/projects/p32737/del6500_home/.conda/envs/cdenv/bin/python /gpfs/projects/p32737/del6500_home/CD/main1.py \
#   --model "$MODEL" \
#   --dataset "$DATASET" \
#   --savepath "/gpfs/projects/p32737/del6500_home/CD/lab_rs" \
#   --use_diag --use_full --use_stacking

# END_TS=$(date +%s)
# RUNTIME=$(( END_TS - START_TS ))
# echo "[$(date)] Job finished."
# echo "Total elapsed time: ${RUNTIME} seconds ≈ $(printf '%.2f' "$(echo "$RUNTIME/3600" | bc -l)") hours"






# module purge
# module load cuda/11.8
# module load anaconda3
# eval "$(conda shell.bash hook)"
# conda activate /gpfs/projects/p32737/del6500_home/.conda/envs/cdenv
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


# cd /gpfs/projects/p32737/del6500_home/CD

# echo "[$(date)] Job started on node $(hostname)"
# START_TS=$(date +%s)

# /gpfs/projects/p32737/del6500_home/.conda/envs/cdenv/bin/python /gpfs/projects/p32737/del6500_home/CD/plot_screen.py

# END_TS=$(date +%s)
# RUNTIME=$(( END_TS - START_TS ))
# echo "[$(date)] Job finished."








set -euxo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/projects/p32737/del6500_home/.conda/envs/cdenv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# 如果你要用站点 CUDA 模块，换成真实存在的版本；否则就不要加载。
# module load cuda/12.1.0-intel

mkdir -p results .hf_cache

echo "=== GPU ==="
srun -n 1 nvidia-smi || true

echo "=== Torch ==="
srun -n 1 python - <<'PY'
import sys, torch, os
print("python:", sys.executable)
print("torch:", torch.__version__, "torch.version.cuda:", torch.version.cuda)
print("built?:", torch.backends.cuda.is_built(), "available?:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available(): print("gpu:", torch.cuda.get_device_name(0))
PY

echo "=== Files ==="
# 确认你要跑的就是“逐层注入”的那份脚本（名字可改，但必须是那份内容）
ls -lh /gpfs/projects/p32737/del6500_home/CD/Steering_generate.py
# 确认16层向量都在
ls -lh lab_rs/samp/google-gemma-2b_STSA_layer{1..16}_stacking_samp.npy

echo "=== RUN per-layer steer (SMOKE TEST) ==="
srun -n 1 python -u /gpfs/projects/p32737/del6500_home/CD/Steering_generate.py \
  --savepath ./lab_rs \
  --model_path ./ \
  --model Qwen/Qwen1.5-14B \
  --quant 16 --cuda 0 \
  --dataset_tag STSA \
  --alpha 0.064 --formula add \
  --num_samples 10 --max_tokens 80 \
  --top_p 0.95 --temperature 0.75 --repetition_penalty 1.1 --no_repeat_ngram_size 3 \
  --out_csv ./results/_smoke_gemma2b_STSA_perlayer.csv \
  --seed 1

echo "=== Check CSV ==="
wc -l ./results/_smoke_gemma2b_STSA_perlayer.csv || true
head -5 ./results/_smoke_gemma2b_STSA_perlayer.csv || true
tail -5 ./results/_smoke_gemma2b_STSA_perlayer.csv || true

echo "=== DONE ==="
