# -*- coding: utf-8 -*-
"""
测试 SFT 微调后的模型
支持交互式对话和批量测试
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel


def load_model(base_model_path: str, lora_path: str = None):
    """
    加载模型
    
    Args:
        base_model_path: 基座模型路径
        lora_path: LoRA adapter 路径 (可选)
    """
    print(f"加载模型...")
    print(f"  基座模型: {base_model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    
    # 加载 LoRA 权重
    if lora_path and os.path.exists(lora_path):
        print(f"  LoRA adapter: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    print("模型加载完成!\n")
    return model, tokenizer


def chat(model, tokenizer, user_input: str, history: list = None, system_prompt: str = None):
    """
    进行对话
    
    Args:
        model: 模型
        tokenizer: tokenizer
        user_input: 用户输入
        history: 对话历史
        system_prompt: 系统提示词
    """
    if history is None:
        history = []
    
    if system_prompt is None:
        system_prompt = "你是一个专业的电商客服助手，负责回答用户关于商品、订单、物流、售后等问题。请用礼貌、专业的语气回答。"
    
    # 构建消息
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加历史
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    
    # 添加当前输入
    messages.append({"role": "user", "content": user_input})
    
    # 应用 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成配置
    generation_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.1,
    )
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    output_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return output_text.strip()


def interactive_chat(base_model_path: str, lora_path: str = None):
    """交互式对话测试"""
    print("=" * 60)
    print("电商客服模型测试 - 交互模式")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清除对话历史")
    print("=" * 60 + "\n")
    
    # 加载模型
    model, tokenizer = load_model(base_model_path, lora_path)
    
    history = []
    
    while True:
        try:
            user_input = input("用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("\n再见!")
                break
            
            if user_input.lower() == "clear":
                history = []
                print("对话历史已清除。\n")
                continue
            
            # 生成回复
            response = chat(model, tokenizer, user_input, history)
            
            # 更新历史
            history.append({"user": user_input, "assistant": response})
            
            print(f"客服: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}\n")


def batch_test(base_model_path: str, lora_path: str = None, test_file: str = None, num_samples: int = 5):
    """批量测试"""
    import json
    
    print("=" * 60)
    print("电商客服模型测试 - 批量模式")
    print("=" * 60 + "\n")
    
    # 加载模型
    model, tokenizer = load_model(base_model_path, lora_path)
    
    # 加载测试数据
    if test_file is None:
        test_file = "/home/rzzhang/llm_sft/data/test.jsonl"
    
    print(f"加载测试数据: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    test_data = test_data[:num_samples]
    print(f"测试样本数: {len(test_data)}\n")
    
    for i, item in enumerate(test_data):
        prompt = item.get("prompt", item.get("input", ""))
        expected = item.get("completion", item.get("output", ""))
        
        # 生成回复
        predicted = chat(model, tokenizer, prompt)
        
        print(f"--- 样本 {i+1} ---")
        print(f"问题: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        print(f"期望: {expected[:200]}{'...' if len(expected) > 200 else ''}")
        print(f"预测: {predicted[:200]}{'...' if len(predicted) > 200 else ''}")
        print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 SFT 微调后的模型")
    parser.add_argument("--mode", type=str, default="chat", choices=["chat", "batch"],
                        help="运行模式: chat (交互对话) 或 batch (批量测试)")
    parser.add_argument("--base_model", type=str, default="/home/rzzhang/models/qwen3.5-2b",
                        help="基座模型路径")
    parser.add_argument("--lora", type=str, default="/home/rzzhang/output/sft",
                        help="LoRA adapter 路径")
    parser.add_argument("--test_file", type=str, default="/home/rzzhang/llm_sft/data/test.jsonl",
                        help="测试数据文件 (batch 模式)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="测试样本数量 (batch 模式)")
    
    args = parser.parse_args()
    
    if args.mode == "chat":
        interactive_chat(args.base_model, args.lora)
    else:
        batch_test(args.base_model, args.lora, args.test_file, args.num_samples)


if __name__ == "__main__":
    main()