# #!/bin/bash
# #SBATCH --account=p32737
# #SBATCH --job-name=gen_text             # 作业名，可自行修改
# #SBATCH --output=/gpfs/projects/p32737/del6500_home/CD/logs/gen_text_%j.out
# #SBATCH --error=/gpfs/projects/p32737/del6500_home/CD/logs/gen_text_%j.err
# #SBATCH --partition=gengpu             # 与之前相同的分区
# #SBATCH --gres=none                   # 若只走 OpenAI API 可改成 --gres=none
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=256G
# #SBATCH --time=08:00:00                # 最长跑 6 小时

# # ======== 环境加载 ========
# module purge
# module load cuda/11.8
# module load anaconda3
# eval "$(conda shell.bash hook)"
# conda activate /gpfs/projects/p32737/del6500_home/.conda/envs/cdenv
# #pip install --quiet --no-cache-dir openai
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# # ======== 切到项目目录 ========
# cd /gpfs/projects/p32737/del6500_home/CD

# # ======== 执行生成脚本 ========
# /gpfs/projects/p32737/del6500_home/.conda/envs/cdenv/bin/python preprocess/generate_text.py\
#     --nthreads 2 \

