# 项目一复现指南：电商客服问答——大模型训练方向

## 项目概述

**目标**：让大模型扛住高频标准问题，把自动解决率从 40% 干到 75% 以上

**技术栈**：
- 基座模型：Qwen3-8B（备选：DeepSeek-R1-Distill-Qwen-7B、Qwen2.5-7B）
- 微调框架：HuggingFace TRL
- 高效微调：HuggingFace PEFT（LoRA/QLoRA）
- 训练基建：PyTorch + DeepSpeed ZeRO-2
- 显卡要求：单卡 3090/4090 可跑 QLoRA

---

## 第一阶段：环境准备（预计 1-2 天）

### 1.1 硬件检查

```bash
# 检查 NVIDIA 显卡
nvidia-smi

# 确认显存 >= 24GB（3090/4090）
# 如果显存不足，需要使用 QLoRA（4bit 量化）
```

### 1.2 创建 Python 虚拟环境

```bash
# 使用 conda 创建环境
conda create -n llm_sft python=3.10 -y
conda activate llm_sft

# 或使用 venv
python -m venv llm_sft_env
source llm_sft_env/bin/activate  # Linux/Mac
# llm_sft_env\Scripts\activate  # Windows
```

### 1.3 安装核心依赖

```bash
# PyTorch（根据 CUDA 版本选择，示例为 CUDA 12.4/12.6）
# 注意：cu124 wheel 支持 CUDA 12.4+，包括 CUDA 12.6
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Transformers 生态（与 PyTorch 2.6.0 兼容）
pip install transformers==4.48.0
pip install accelerate==0.34.0
pip install peft==0.14.0
pip install bitsandbytes==0.45.0  # QLoRA 需要，支持 CUDA 12

# TRL 训练框架
pip install trl==0.14.0

# 数据处理
pip install datasets pandas numpy

# DeepSpeed（可选，多卡或大模型需要）
pip install deepspeed

# 评估工具
pip install evaluate rouge-score

# 其他工具
pip install tqdm wandb tensorboard
```

### 1.4 验证安装

```python
# test_install.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

---

## 第二阶段：数据准备（预计 3-4 天）

### 2.1 数据来源

**方式一：GitHub 开源数据集**

搜索关键词：`e-commerce customer service dataset`

推荐数据集：
- https://huggingface.co/datasets/silk-road/Chat-Instruction-PT（中文客服对话）
- https://github.com/Paphi/Dialogue-Datasets（多轮对话数据）

**方式二：电商平台 FAQ 抓取**

```python
# 示例：从淘宝"问大家"板块整理数据
# 手动收集商品问答，整理成结构化格式
```

**方式三：GPT-4 扩写（推荐）**

```python
# templates.py
import json
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# 客服场景模板
SCENARIOS = {
    "退换货": [
        "这个商品能退吗？已经拆封了",
        "买了7天还能无理由退货吗？",
        "退货运费谁承担？",
    ],
    "物流": [
        "我的快递到哪了？",
        "可以加急发货吗？",
        "能改收货地址吗？",
    ],
    "商品参数": [
        "这款衣服是什么材质的？",
        "这个手机支持5G吗？",
        "保质期多长时间？",
    ],
    "投诉": [
        "客服态度太差了要投诉",
        "发错货了怎么处理？",
        "承诺的赠品没收到",
    ]
}

def generate_instruction_data(query, scenario):
    """使用 GPT-4 生成客服回答"""
    prompt = f"""你是一名专业的电商客服，请针对以下用户问题给出专业、礼貌、有帮助的回答。
    
用户问题：{query}
场景分类：{scenario}

要求：
1. 回答要准确、专业
2. 语气要礼貌、有亲和力
3. 给出具体的解决方案或建议
4. 控制在100字以内

请直接给出回答："""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    
    return response.choices[0].message.content

# 批量生成数据
dataset = []
for scenario, queries in SCENARIOS.items():
    for query in queries:
        answer = generate_instruction_data(query, scenario)
        dataset.append({
            "instruction": f"用户问：{query}",
            "input": f"场景：{scenario}",
            "output": answer
        })

# 保存
with open("ecommerce_cs_data.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
```

### 2.2 数据清洗

```python
# data_cleaning.py
import json
import pandas as pd
from collections import Counter

def clean_ecommerce_data(input_file, output_file):
    """清洗电商客服数据"""
    
    # 加载原始数据
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"原始数据量: {len(raw_data)}")
    
    cleaned_data = []
    seen_queries = set()  # 去重
    
    for item in raw_data:
        query = item["instruction"]
        answer = item["output"]
        
        # 1. 去重
        if query in seen_queries:
            continue
        seen_queries.add(query)
        
        # 2. 去短回复（<20字）
        if len(answer) < 20:
            continue
        
        # 3. 去矛盾答案（需要人工规则或用模型判断）
        # 这里简单示例：去掉包含"不知道"或"不清楚"的回答
        if "不知道" in answer or "不清楚" in answer:
            continue
        
        # 4. 格式化
        cleaned_data.append({
            "instruction": query,
            "input": item.get("input", ""),
            "output": answer
        })
    
    print(f"清洗后数据量: {len(cleaned_data)}")
    
    # 保存清洗后数据
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    # 记录清洗统计
    stats = {
        "原始数量": len(raw_data),
        "去重数量": len(raw_data) - len(seen_queries),
        "短回复数量": sum(1 for item in raw_data if len(item["output"]) < 20),
        "最终数量": len(cleaned_data)
    }
    print("清洗统计:", stats)
    
    return cleaned_data

# 执行清洗
clean_ecommerce_data("ecommerce_cs_data.json", "ecommerce_cs_data_cleaned.json")
```

### 2.3 转换为训练格式

```python
# convert_to_train_format.py
import json
from datasets import Dataset

def convert_to_sft_format(input_file, output_file):
    """转换为 TRL SFT 训练格式"""
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # 构建完整的 prompt
        prompt = f"{item['instruction']}\n{item['input']}" if item['input'] else item['instruction']
        response = item['output']
        
        # TRL SFT 格式：一个 text 字段包含完整对话
        # 或者使用 prompt/completion 格式
        formatted_data.append({
            "prompt": prompt,
            "completion": response,
            # 或者用一个 text 字段
            "text": f"### 指令:\n{prompt}\n\n### 回答:\n{response}"
        })
    
    # 保存为 JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 创建 HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)
    dataset.save_to_disk("ecommerce_cs_dataset")
    
    print(f"转换完成，共 {len(formatted_data)} 条数据")
    return dataset

convert_to_sft_format("ecommerce_cs_data_cleaned.json", "train_data.jsonl")
```

### 2.4 数据划分

```python
# split_dataset.py
from datasets import load_dataset

# 加载数据
dataset = load_dataset("json", data_files="train_data.jsonl")

# 划分 train/val/test (80/10/10)
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
test_val = split_dataset["test"].train_test_split(test_size=0.5, seed=42)

final_dataset = {
    "train": split_dataset["train"],
    "validation": test_val["train"],
    "test": test_val["test"]
}

# 保存
from datasets import DatasetDict
dataset_dict = DatasetDict(final_dataset)
dataset_dict.save_to_disk("ecommerce_cs_final_dataset")

print(f"训练集: {len(final_dataset['train'])} 条")
print(f"验证集: {len(final_dataset['validation'])} 条")
print(f"测试集: {len(final_dataset['test'])} 条")
```

---

## 第三阶段：SFT 训练（第1周）

### 3.1 下载基座模型

```python
# download_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或 Qwen/Qwen3-8B

# 下载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.save_pretrained("./models/Qwen2.5-7B-Instruct")

# 下载模型（可选：使用 bitsandbytes 量化加载减少显存）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
model.save_pretrained("./models/Qwen2.5-7B-Instruct")

print("模型下载完成")
```

### 3.2 SFT 训练脚本

```python
# sft_train.py
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import wandb

# ============ 配置 ============
MODEL_PATH = "./models/Qwen2.5-7B-Instruct"
DATA_PATH = "./ecommerce_cs_final_dataset"
OUTPUT_DIR = "./output/sft_qwen_ecommerce"

# LoRA 配置
LORA_CONFIG = {
    "r": 16,  # rank
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM
}

# 训练配置
TRAIN_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "save_total_limit": 3,
    "fp16": True,
    "gradient_checkpointing": True,
    "max_seq_length": 512
}

# ============ 加载数据 ============
print("加载数据集...")
dataset = load_from_disk(DATA_PATH)
print(f"训练集: {len(dataset['train'])} 条")
print(f"验证集: {len(dataset['validation'])} 条")

# ============ 加载模型 ============
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ============ 配置 LoRA ============
print("配置 LoRA...")
peft_config = LoraConfig(**LORA_CONFIG)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ============ 训练参数 ============
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    **TRAIN_CONFIG,
    report_to="wandb"
)

# ============ 开始训练 ============
print("开始 SFT 训练...")
wandb.init(project="ecommerce-cs-sft", name="qwen-lora-r16")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    max_seq_length=TRAIN_CONFIG["max_seq_length"],
    packing=False
)

trainer.train()

# ============ 保存模型 ============
print("保存模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

wandb.finish()
print("训练完成！")
```

### 3.3 QLoRA 训练（显存不足时使用）

```python
# qlora_train.py
import torch
from transformers import BitsAndBytesConfig

# QLoRA 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 4bit 量化加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto"
)

# LoRA 配置（QLoRA）
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 其余训练代码与 SFT 相同
```

### 3.4 训练监控

```bash
# 启动 TensorBoard
tensorboard --logdir ./output/sft_qwen_ecommerce

# 或使用 wandb（需要在代码中配置）
wandb login
```

---

## 第四阶段：LoRA 消融实验（第2周）

### 4.1 Rank 消融实验

```python
# ablation_rank.py
import itertools
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer

# 消融配置
RANK_VALUES = [4, 16, 64]
BASE_CONFIG = {
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "task_type": TaskType.CAUSAL_LM
}

results = []

for rank in RANK_VALUES:
    print(f"\n{'='*50}")
    print(f"Running experiment: rank={rank}")
    print(f"{'='*50}\n")
    
    # 配置
    peft_config = LoraConfig(r=rank, lora_alpha=rank*2, **BASE_CONFIG)
    output_dir = f"./output/ablation_rank_{rank}"
    
    # 训练（使用相同的训练配置）
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        # ... 其他配置
    )
    
    # ... 训练代码 ...
    
    # 评估并记录
    eval_result = trainer.evaluate()
    
    results.append({
        "rank": rank,
        "eval_loss": eval_result["eval_loss"],
        "memory_used": torch.cuda.max_memory_allocated() / 1024**3
    })

# 打印结果对比
import pandas as pd
df = pd.DataFrame(results)
print("\n消融实验结果:")
print(df.to_markdown())
```

### 4.2 Target Modules 消融

```python
# ablation_target_modules.py
TARGET_MODULES_OPTIONS = [
    ["q_proj", "v_proj"],  # 最小配置
    ["q_proj", "k_proj", "v_proj", "o_proj"],  # 全 attention
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 全线性层
]

# 类似 rank 消融的实验代码
```

### 4.3 QLoRA vs LoRA 对比

```python
# ablation_quantization.py
QUANTIZATION_OPTIONS = ["fp16", "8bit", "4bit"]

# 对比实验：精度、显存、训练时间
```

---

## 第五阶段：DPO 偏好对齐（第3周）

### 5.1 构建偏好数据

```python
# build_preference_data.py
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 SFT 后的模型
MODEL_PATH = "./output/sft_qwen_ecommerce"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_responses(prompt, num_responses=3):
    """生成多个候选回答"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    responses = []
    for i in range(num_responses):
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7 + i * 0.2,  # 不同温度
            do_sample=True,
            top_p=0.9
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        responses.append(response)
    
    return responses

# 加载测试问题
with open("test_queries.json", "r", encoding="utf-8") as f:
    test_queries = json.load(f)

# 生成偏好数据
preference_data = []

for query in test_queries:
    responses = generate_responses(query["prompt"], num_responses=3)
    
    # 人工标注或使用规则自动标注
    # 这里示例：人工选择
    print(f"\n问题: {query['prompt']}")
    for i, resp in enumerate(responses):
        print(f"\n回答 {i+1}: {resp}")
    
    # 选择最佳和最差
    chosen_idx = int(input("选择最佳回答 (1-3): ")) - 1
    rejected_idx = int(input("选择最差回答 (1-3): ")) - 1
    
    preference_data.append({
        "prompt": query["prompt"],
        "chosen": responses[chosen_idx],
        "rejected": responses[rejected_idx]
    })

# 保存偏好数据
with open("preference_data.json", "w", encoding="utf-8") as f:
    json.dump(preference_data, f, ensure_ascii=False, indent=2)
```

### 5.2 DPO 训练

```python
# dpo_train.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import PeftModel

# ============ 配置 ============
SFT_MODEL_PATH = "./output/sft_qwen_ecommerce"
PREFERENCE_DATA_PATH = "./preference_data.json"
OUTPUT_DIR = "./output/dpo_qwen_ecommerce"

# ============ 加载模型 ============
# 加载 SFT 后的模型作为 policy model
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
policy_model = AutoModelForCausalLM.from_pretrained(
    SFT_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载原始模型作为 reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# ============ 加载偏好数据 ============
preference_dataset = load_dataset("json", data_files=PREFERENCE_DATA_PATH)

# ============ DPO 训练配置 ============
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.1,  # DPO 温度参数
    num_train_epochs=2,
    per_device_train_batch_size=2,
    learning_rate=5e-6,
    gradient_accumulation_steps=4,
    logging_steps=10,
    save_steps=100,
    fp16=True
)

# ============ 开始 DPO 训练 ============
dpo_trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset["train"],
    tokenizer=tokenizer
)

dpo_trainer.train()

# 保存
dpo_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

### 5.3 DPO β 参数消融

```python
# dpo_beta_ablation.py
BETA_VALUES = [0.05, 0.1, 0.3]

for beta in BETA_VALUES:
    dpo_config = DPOConfig(
        beta=beta,
        # ... 其他配置
    )
    # 运行 DPO 训练并评估
```

---

## 第六阶段：模型评估（第4周）

### 6.1 离线评估脚本

```python
# evaluate.py
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def evaluate_model(model_path, test_data_path):
    """评估模型在测试集上的表现"""
    
    tokenizer, model = load_model(model_path)
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    results = []
    
    for item in tqdm(test_data):
        prompt = item["prompt"]
        ground_truth = item["output"]
        
        # 生成回答
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = prediction.replace(prompt, "").strip()
        
        # 计算指标
        results.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "prediction": prediction,
            # 可以添加更多指标
        })
    
    # 保存结果
    with open(f"eval_results_{model_path.split('/')[-1]}.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

# 对比 SFT 和 DPO 模型
evaluate_model("./output/sft_qwen_ecommerce", "test_data.json")
evaluate_model("./output/dpo_qwen_ecommerce", "test_data.json")
```

### 6.2 BadCase 分析

```python
# badcase_analysis.py
import json
from collections import Counter

def analyze_badcases(results_path):
    """分析失败案例"""
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    badcases = []
    
    for item in results:
        prediction = item["prediction"]
        ground_truth = item["ground_truth"]
        
        # 识别问题类型
        issues = []
        
        # 1. 复读机问题
        if prediction.count("您好") > 3:
            issues.append("复读机")
        
        # 2. 太短
        if len(prediction) < 20:
            issues.append("回复过短")
        
        # 3. 答非所问（简单检测）
        if "退货" in item["prompt"] and "退货" not in prediction:
            issues.append("可能答非所问")
        
        # 4. 事实错误（需要更复杂的检测）
        # ...
        
        if issues:
            badcases.append({
                "prompt": item["prompt"],
                "prediction": prediction,
                "ground_truth": ground_truth,
                "issues": issues
            })
    
    # 统计问题类型
    issue_counter = Counter()
    for case in badcases:
        for issue in case["issues"]:
            issue_counter[issue] += 1
    
    print("问题类型统计:")
    for issue, count in issue_counter.most_common():
        print(f"  {issue}: {count} 次")
    
    return badcases

badcases = analyze_badcases("eval_results_sft.json")
```

---

## 第七阶段：推理部署

### 7.1 合并 LoRA 权重

```python
# merge_lora.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen2.5-7B-Instruct")

# 加载 LoRA 权重
model = PeftModel.from_pretrained(
    base_model,
    "./output/dpo_qwen_ecommerce"
)

# 合并权重
merged_model = model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("./output/merged_ecommerce_model")
tokenizer.save_pretrained("./output/merged_ecommerce_model")

print("LoRA 权重合并完成")
```

### 7.2 使用 vLLM 部署

```bash
# 安装 vLLM
pip install vllm

# 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model ./output/merged_ecommerce_model \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048
```

### 7.3 测试 API

```python
# test_api.py
import requests
import json

def chat(query):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "./output/merged_ecommerce_model",
            "messages": [
                {"role": "user", "content": query}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
    )
    return response.json()

# 测试
result = chat("这个商品能退吗？已经拆封了")
print(result["choices"][0]["message"]["content"])
```

---

## 项目文件结构

```
ecommerce_cs_project/
├── data/
│   ├── raw/                      # 原始数据
│   ├── cleaned/                  # 清洗后数据
│   ├── train.jsonl               # 训练数据
│   ├── val.jsonl                 # 验证数据
│   └── test.jsonl                # 测试数据
├── models/
│   └── Qwen2.5-7B-Instruct/      # 基座模型
├── output/
│   ├── sft_qwen_ecommerce/       # SFT 输出
│   ├── dpo_qwen_ecommerce/       # DPO 输出
│   └── merged_ecommerce_model/   # 合并后模型
├── scripts/
│   ├── data_cleaning.py
│   ├── sft_train.py
│   ├── dpo_train.py
│   ├── evaluate.py
│   └── merge_lora.py
├── logs/
│   └── tensorboard/
└── docs/
    ├── experiment_results.md     # 实验结果
    └── badcase_analysis.md       # BadCase 分析
```

---

## 实验结果记录模板

### 消融实验结果表

| 实验 | 配置 | Eval Loss | 准确率 | 显存占用 | 训练时间 |
|------|------|-----------|--------|----------|----------|
| rank=4 | LoRA, r=4 | ? | ? | ? | ? |
| rank=16 | LoRA, r=16 | ? | ? | ? | ? |
| rank=64 | LoRA, r=64 | ? | ? | ? | ? |
| QLoRA | 4bit + LoRA r=16 | ? | ? | ? | ? |

### DPO 结果对比

| 配置 | 偏好胜率 | 准确率 | 多样性 |
|------|----------|--------|--------|
| SFT-only | - | ? | - |
| SFT + DPO (β=0.05) | ? | ? | ? |
| SFT + DPO (β=0.1) | ? | ? | ? |
| SFT + DPO (β=0.3) | ? | ? | ? |

---

## 简历写法参考

> 面向电商客服智能问答场景，针对退换货、物流、商品咨询等高频问题，基于 Qwen2.5-7B 完成指令微调与偏好对齐全流程。清洗并构建 5800 条客服指令数据集完成 LoRA SFT，对 rank、target_modules、量化方案进行消融实验；基于客服满意度维度标注 2000 组偏好数据完成 DPO 对齐。最终回答准确率从 58% 提升至 82%，用户偏好胜率达 67%，自动解决率由 40% 提升至 72%，并沉淀 badcase 分析与迭代策略。

---

## 面试高频问题准备

1. **LoRA 的 rank 怎么选的？**
   - 讲消融实验结论：rank=16 在效果和效率上达到最优平衡

2. **DPO 和 PPO 为什么选 DPO？**
   - DPO 不需要额外训练 Reward Model
   - 训练更稳定，不易发散
   - 数据量要求相对较低

3. **loss 不降怎么排查？**
   - 学习率调整（太大/太小）
   - 数据质量问题
   - 梯度裁剪配置
   - 混合精度配置

4. **β 调大调小会怎样？**
   - β 大：更保守，不偏离参考模型
   - β 小：偏好更强，但易过拟合

5. **出现复读机怎么办？**
   - 检查训练数据中的重复句式
   - 清洗数据
   - 调整温度参数

---

## 相关开源仓库

- **LLaMA-Factory**：https://github.com/hiyouga/LLaMA-Factory （SFT+DPO 全流程参考）
- **TRL**：https://github.com/huggingface/trl （SFT/DPO 训练框架）
- **PEFT**：https://github.com/huggingface/peft （LoRA/QLoRA）
- **Transformers**：https://github.com/huggingface/transformers
- **Qwen**：https://github.com/QwenLM/Qwen2.5 （基座模型）