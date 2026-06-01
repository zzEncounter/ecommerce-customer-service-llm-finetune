# -*- coding: utf-8 -*-
"""
模型推理脚本
支持加载 SFT/DPO 训练后的模型进行对话
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel


class EcommerceChatBot:
    """电商客服聊天机器人"""
    
    def __init__(
        self,
        model_path: str = "/home/rzzhang/output/sft",
        base_model: str = "/home/rzzhang/models/qwen3.5-2b",
        device: str = "auto",
    ):
        """
        初始化聊天机器人
        
        Args:
            model_path: 微调模型路径 (LoRA 权重)
            base_model: 基座模型路径
            device: 设备类型
        """
        self.model_path = model_path
        self.base_model = base_model
        
        print(f"加载模型...")
        print(f"  基座模型: {base_model}")
        print(f"  LoRA 权重: {model_path if os.path.exists(model_path) else '无'}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            device_map=device,
        )
        
        # 加载 LoRA 权重（如果存在）
        if os.path.exists(model_path):
            print(f"  加载 LoRA 权重...")
            self.model = PeftModel.from_pretrained(self.model, model_path)
        
        # 生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
        )
        
        # 系统提示词
        self.system_prompt = "你是一个专业的电商客服助手，负责回答用户关于商品、订单、物流、售后等问题。请用礼貌、专业的语气回答。"
        
        # 对话历史
        self.history = []
        
        print("模型加载完成!\n")
    
    def chat(self, user_input: str, clear_history: bool = False) -> str:
        """
        进行对话
        
        Args:
            user_input: 用户输入
            clear_history: 是否清除历史
        
        Returns:
            模型回复
        """
        if clear_history:
            self.history = []
        
        # 构建消息
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        # 添加历史
        for turn in self.history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        
        # 添加当前输入
        messages.append({"role": "user", "content": user_input})
        
        # 应用 chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        output_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        # 更新历史
        self.history.append({
            "user": user_input,
            "assistant": output_text.strip(),
        })
        
        return output_text.strip()
    
    def clear_history(self):
        """清除对话历史"""
        self.history = []


def interactive_chat(model_path: str = "/home/rzzhang/output/sft"):
    """交互式对话"""
    print("="*60)
    print("电商客服聊天机器人")
    print("="*60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清除对话历史")
    print("="*60 + "\n")
    
    # 初始化机器人
    bot = EcommerceChatBot(model_path=model_path)
    
    while True:
        try:
            user_input = input("用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("\n再见!")
                break
            
            if user_input.lower() == "clear":
                bot.clear_history()
                print("对话历史已清除。\n")
                continue
            
            # 生成回复
            response = bot.chat(user_input)
            print(f"客服: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}\n")


def batch_inference(
    model_path: str = "/home/rzzhang/output/sft",
    input_file: str = "/home/rzzhang/llm_sft/data/test.jsonl",
    output_file: str = "/home/rzzhang/output/predictions.json",
    num_samples: int = 10,
):
    """批量推理"""
    import json
    
    print("="*60)
    print("批量推理")
    print("="*60)
    
    # 初始化机器人
    bot = EcommerceChatBot(model_path=model_path)
    
    # 加载测试数据
    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    # 限制数量
    test_data = test_data[:num_samples]
    
    results = []
    for i, item in enumerate(test_data):
        prompt = item["prompt"]
        expected = item["completion"]
        
        # 生成回复
        predicted = bot.chat(prompt, clear_history=True)
        
        results.append({
            "id": i,
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
        })
        
        print(f"\n--- 样本 {i+1} ---")
        print(f"输入: {prompt[:100]}...")
        print(f"期望: {expected[:100]}...")
        print(f"预测: {predicted[:100]}...")
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="电商客服模型推理")
    parser.add_argument("--mode", type=str, default="chat", choices=["chat", "batch"],
                        help="运行模式: chat (交互对话) 或 batch (批量推理)")
    parser.add_argument("--model_path", type=str, default="/home/rzzhang/output/sft",
                        help="模型路径")
    parser.add_argument("--input_file", type=str, default="/home/rzzhang/llm_sft/data/test.jsonl",
                        help="批量推理输入文件")
    parser.add_argument("--output_file", type=str, default="/home/rzzhang/output/predictions.json",
                        help="批量推理输出文件")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="批量推理样本数量")
    
    args = parser.parse_args()
    
    if args.mode == "chat":
        interactive_chat(model_path=args.model_path)
    else:
        batch_inference(
            model_path=args.model_path,
            input_file=args.input_file,
            output_file=args.output_file,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()