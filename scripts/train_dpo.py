# -*- coding: utf-8 -*-
"""
DPO (Direct Preference Optimization) 偏好对齐训练脚本
使用电商客服数据中的正负样本构建偏好对
"""
import os
import torch
from dataclasses import dataclass, field
import json
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import PeftModel, LoraConfig, get_peft_model


@dataclass
class DPOArguments:
    """DPO 参数"""
    model_name_or_path: str = field(
        default="/home/rzzhang/output/sft",
        metadata={"help": "SFT 模型路径"}
    )
    data_dir: str = field(
        default="/home/rzzhang/llm_sft/data",
        metadata={"help": "数据目录"}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "DPO beta 参数"}
    )
    learning_rate: float = field(
        default=5e-6,
        metadata={"help": "学习率"}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "训练轮数"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大长度"}
    )
    max_prompt_length: int = field(
        default=256,
        metadata={"help": "最大 prompt 长度"}
    )


def build_preference_pairs(data_dir, raw_data_dir):
    """
    从原始数据构建偏好对 (chosen vs rejected)
    
    原始数据中:
    - label=1 是正确的回复 (chosen)
    - label=0 是错误的回复 (rejected)
    
    需要将相同对话上下文的正负样本配对
    """
    print("构建偏好对...")
    
    def load_raw_data(filepath):
        """加载原始数据"""
        data_by_context = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载数据"):
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                label = parts[0]
                conversation_parts = parts[1:-1]
                response = parts[-1]
                
                # 构建对话上下文
                context = '\t'.join(conversation_parts)
                
                if context not in data_by_context:
                    data_by_context[context] = {"chosen": [], "rejected": []}
                
                if label == '1':
                    data_by_context[context]["chosen"].append(response)
                else:
                    data_by_context[context]["rejected"].append(response)
        
        return data_by_context
    
    # 加载训练数据
    raw_train_path = os.path.join(raw_data_dir, "train.txt")
    data_by_context = load_raw_data(raw_train_path)
    
    # 构建偏好对
    preference_pairs = []
    for context, responses in data_by_context.items():
        chosen_list = responses["chosen"]
        rejected_list = responses["rejected"]
        
        # 为每个 chosen 找一个 rejected 配对
        for chosen in chosen_list:
            if rejected_list:
                # 简单地取第一个 rejected
                rejected = rejected_list[0]
                
                # 构建对话格式
                conversation_parts = context.split('\t')
                conversation_text = ""
                for i, utterance in enumerate(conversation_parts):
                    if i % 2 == 0:
                        conversation_text += f"用户: {utterance}\n"
                    else:
                        conversation_text += f"客服: {utterance}\n"
                
                preference_pairs.append({
                    "prompt": conversation_text.strip(),
                    "chosen": chosen,
                    "rejected": rejected,
                })
    
    print(f"构建了 {len(preference_pairs)} 个偏好对")
    
    # 限制数量（DPO 训练数据量不需要太大）
    max_pairs = 10000
    if len(preference_pairs) > max_pairs:
        preference_pairs = preference_pairs[:max_pairs]
        print(f"限制为 {max_pairs} 个偏好对")
    
    return preference_pairs


def main():
    """主训练函数"""
    print("="*60)
    print("DPO 偏好对齐训练 - 电商客服模型")
    print("="*60)
    
    # 参数设置
    dpo_args = DPOArguments()
    
    # 原始数据目录
    raw_data_dir = "/home/rzzhang/llm_sft/E-commerce dataset/E-commerce dataset"
    
    # 检查 SFT 模型是否存在
    sft_model_path = dpo_args.model_name_or_path
    base_model_path = "/home/rzzhang/models/qwen3.5-2b"
    
    if not os.path.exists(sft_model_path):
        print(f"警告: SFT 模型不存在: {sft_model_path}")
        print(f"将使用基座模型: {base_model_path}")
        sft_model_path = base_model_path
    
    print(f"\n模型路径: {sft_model_path}")
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 加载 tokenizer
    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    
    # 如果有 SFT 模型，加载 LoRA 权重（检查 adapter_config.json）
    lora_config_path = os.path.join(sft_model_path, "adapter_config.json")
    if os.path.exists(lora_config_path):
        print("加载 SFT LoRA 权重...")
        model = PeftModel.from_pretrained(model, sft_model_path)
        print("合并 SFT LoRA 权重到基座模型...")
        model = model.merge_and_unload()
        print("SFT LoRA 权重已合并，模型现在是完整的 SFT 模型")
    else:
        print("不加载 LoRA 权重（仅使用基座模型）")
    
    # 为 DPO 训练创建新的 LoRA 适配器
    print("\n创建 DPO LoRA 适配器...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 构建偏好对
    preference_pairs = build_preference_pairs(dpo_args.data_dir, raw_data_dir)
    
    # 创建数据集
    dataset = Dataset.from_list(preference_pairs)
    
    # 划分训练和验证
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"训练数据: {len(train_dataset)} 条")
    print(f"验证数据: {len(val_dataset)} 条")
    
    # 定义 DPO 配置 (DPOConfig 继承自 TrainingArguments，所有参数放在一起)
    training_args = DPOConfig(
        output_dir="/home/rzzhang/output/dpo",
        num_train_epochs=dpo_args.num_train_epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=dpo_args.learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=False,  # 禁用梯度检查点，避免与 PEFT/LoRA 冲突
        # DPO 专属参数
        max_length=dpo_args.max_length,
        beta=dpo_args.beta,
    )
    
    # 在开始训练前关闭 cache
    model.config.use_cache = False

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer, 
    )
    
    # 开始训练
    print("\n开始 DPO 训练...")
    trainer.train()
    
    # 保存模型
    print("\n保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"\n模型已保存到: {training_args.output_dir}")
    print("DPO 训练完成!")


if __name__ == "__main__":
    main()