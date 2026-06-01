#!/bin/bash

# ================= 资源配置 =================
#SBATCH --job-name=llm_sft           # 作业名称
#SBATCH --partition=a100             # 分区：思源一号A100集群
#SBATCH -N 1                         # 节点数：1台机器
#SBATCH --ntasks-per-node=1          # 任务数：1个主进程
#SBATCH --cpus-per-task=16           # CPU核数：A100通常配16核
#SBATCH --gres=gpu:1                 # GPU资源：申请1张显卡
#SBATCH --output=%j.out              # 标准输出日志 (如 123456.out)
#SBATCH --error=%j.err               # 错误日志文件
#SBATCH --mail-type=end              # 结束发送邮件通知
#SBATCH --mail-user=2267365907@qq.com    # 【记得改成你的邮箱】

# ================= 环境加载 =================

# 1. 加载系统基础模块
# 注意：思源一号可能需要显式加载 cuda 模块来支持底层驱动

# 2. 激活 Conda 虚拟环境
# 关键步骤：先 source bashrc 确保 conda 命令可用
source ~/.bashrc
conda activate llm_sft

# ================= 调试信息 (可选) =================
# 运行前检查一下 Python 和 GPU 状态，确保没找错环境
echo "=== Environment Check ==="
which python
nvidia-smi
echo "========================="

# ================= 执行任务 =================
# 使用绝对路径运行你的 Python 脚本
python /dssg/home/acct-anbangwu/zhang-rz/sft_project/llm/test.py


