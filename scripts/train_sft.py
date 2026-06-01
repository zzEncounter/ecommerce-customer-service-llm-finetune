# -*- coding: utf-8 -*-
"""
SFT (Supervised Fine-Tuning) 训练脚本
使用 Qwen2.5-1.5B-Instruct 作为基座模型
"""
print(">>> script start")
import os
import torch
print("import torch done")
from dataclasses import dataclass, field
from typing import Optional
import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
print("import transformers done")
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
#import transformers



@dataclass
class ModelArguments:
    """模型参数"""
    #model_name_or_path: str = field(
    #    default="Qwen/Qwen2.5-1.5B-Instruct",
    #    metadata={"help": "基座模型路径"}
    #)
    model_name_or_path = "/home/rzzhang/models/qwen3.5-2b"
    use_lora: bool = field(
        default=True,
        metadata={"help": "是否使用 LoRA"}
    )
    lora_rank: int = field(
        default=4,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )


@dataclass
class DataArguments:
    """数据参数"""
    data_dir: str = field(
        default="/home/rzzhang/llm_sft/data",
        metadata={"help": "数据目录"}
    )
    max_source_length: int = field(
        default=512,
        metadata={"help": "最大输入长度"}
    )
    max_target_length: int = field(
        default=128,
        metadata={"help": "最大输出长度"}
    )


def load_dataset(data_dir):
    """加载训练数据"""
    def load_jsonl(filepath):
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    train_data = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    val_data = load_jsonl(os.path.join(data_dir, "val.jsonl"))
    
    print(f"训练数据: {len(train_data)} 条")
    print(f"验证数据: {len(val_data)} 条")
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset


def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    """数据预处理 - 将整个序列一起处理"""
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    
    max_length = max_source_length + max_target_length
    
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        # 构建输入部分（system + user）
        input_messages = [
            {"role": "system", "content": "你是一个专业的电商客服助手，负责回答用户关于商品、订单、物流、售后等问题。请用礼貌、专业的语气回答。"},
            {"role": "user", "content": prompt},
        ]
        
        # 获取输入部分的文本
        input_text = tokenizer.apply_chat_template(
            input_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 构建完整序列（输入 + 输出）
        full_text = input_text + completion + tokenizer.eos_token
        
        # 对整个序列 tokenize
        tokenized = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        # 对输入部分 tokenize 以获取长度
        input_tokenized = tokenizer(
            input_text,
            max_length=max_source_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # 创建 labels：输入部分设为 -100，输出部分保持原值
        labels = [-100] * len(input_ids)
        input_len = len(input_tokenized["input_ids"])
        
        # 将输出部分的 labels 设为实际的 token ids
        for i in range(input_len, len(input_ids)):
            labels[i] = input_ids[i]
        
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)
    
    return model_inputs


def main():
    """主训练函数"""
    print("="*60)
    print("SFT 训练 - 电商客服模型")
    print("="*60)
    
    # 参数设置
    model_args = ModelArguments()
    data_args = DataArguments()
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="/home/rzzhang/output/sft_rank4",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )
    
    print(f"\n模型: {model_args.model_name_or_path}")
    print(f"使用 LoRA: {model_args.use_lora}")
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 加载 tokenizer
    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    
    # LoRA 配置
    if model_args.use_lora:
        print("配置 LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 禁用 cache 以兼容 gradient checkpointing
    model.config.use_cache = False
    
    # 启用输入梯度（gradient checkpointing 需要）
    model.enable_input_require_grads()
    
    # 加载数据
    print("\n加载数据...")
    train_dataset, val_dataset = load_dataset(data_args.data_dir)
    
    # 数据预处理
    print("预处理数据...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, data_args.max_source_length, data_args.max_target_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="预处理训练数据",
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, data_args.max_source_length, data_args.max_target_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="预处理验证数据",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n开始训练...")
    trainer.train()
    
    # 保存模型
    print("\n保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"\n模型已保存到: {training_args.output_dir}")
    print("训练完成!")


if __name__ == "__main__":
    main()