# -*- coding: utf-8 -*-
"""
模型评估脚本
计算 BLEU、ROUGE 等指标
"""
import os
import json
import torch
from tqdm import tqdm
from collections import Counter

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel


def calculate_bleu(reference: str, candidate: str, n: int = 4) -> float:
    """计算 BLEU 分数"""
    from collections import Counter
    import math
    
    def ngrams(text, n):
        tokens = list(text)
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    if not candidate or not reference:
        return 0.0
    
    # 计算 n-gram 精度
    precisions = []
    for i in range(1, n+1):
        ref_ngrams = Counter(ngrams(reference, i))
        cand_ngrams = Counter(ngrams(candidate, i))
        
        # 计算匹配数量
        matches = 0
        for ngram, count in cand_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
        
        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(matches / total)
    
    # 计算 brevity penalty
    ref_len = len(reference)
    cand_len = len(candidate)
    
    if cand_len == 0:
        return 0.0
    
    if cand_len >= ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / cand_len)
    
    # 计算 BLEU
    if min(precisions) == 0:
        return 0.0
    
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / n)
    return bleu


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """计算 ROUGE-L 分数"""
    def lcs(s1, s2):
        """最长公共子序列"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    if not candidate or not reference:
        return 0.0
    
    lcs_len = lcs(list(reference), list(candidate))
    
    # F1 score
    precision = lcs_len / len(candidate) if candidate else 0
    recall = lcs_len / len(reference) if reference else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_model(
    model_path: str = "/home/rzzhang/output/sft",
    test_file: str = "/home/rzzhang/llm_sft/data/test.jsonl",
    output_file: str = "/home/rzzhang/output/evaluation_results.json",
    max_samples: int = None,
    sft_model_path: str = "/home/rzzhang/output/sft",
):
    """评估模型
    
    Args:
        model_path: 要评估的模型路径
        test_file: 测试数据文件
        output_file: 输出文件
        max_samples: 最大样本数
        sft_model_path: SFT 模型路径（用于 DPO 模型评估时先加载 SFT 权重）
    """
    print("="*60)
    print("模型评估")
    print("="*60)
    
    # 加载测试数据
    print(f"\n加载测试数据: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    print(f"测试样本数: {len(test_data)}")
    
    # 加载模型
    base_model = "/home/rzzhang/models/qwen3.5-2b"
    print(f"\n加载模型...")
    print(f"  基座模型: {base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
    )
    
    # 判断是否加载 LoRA 权重：检查路径是否存在且包含 adapter_config.json
    lora_config_path = os.path.join(model_path, "adapter_config.json") if model_path else None
    if model_path and os.path.exists(lora_config_path):
        # 检查是否是 DPO 模型（需要在 SFT 基础上加载）
        # DPO 模型的 base_model_name_or_path 是基座模型，而不是 SFT 模型
        is_dpo_model = "dpo" in model_path.lower()
        
        if is_dpo_model:
            print(f"  检测到 DPO 模型，先加载 SFT 权重...")
            sft_lora_path = os.path.join(sft_model_path, "adapter_config.json")
            if os.path.exists(sft_lora_path):
                model = PeftModel.from_pretrained(model, sft_model_path)
                print(f"    SFT 权重已加载: {sft_model_path}")
                # 关键：合并 SFT LoRA 到基座模型，然后再加载 DPO LoRA
                print(f"    合并 SFT LoRA 到基座模型...")
                model = model.merge_and_unload()
                print(f"    SFT 权重已合并到基座模型")
            else:
                print(f"    警告: SFT 模型不存在: {sft_model_path}")
        
        print(f"  加载目标 LoRA 权重: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        print(f"  LoRA 权重加载完成")
    else:
        print(f"  不加载 LoRA 权重（仅使用基座模型）")
    
    model.eval()
    
    # 评估
    results = []
    all_bleu = []
    all_rouge = []
    
    system_prompt = "你是一个专业的电商客服助手，负责回答用户关于商品、订单、物流、售后等问题。"
    
    print("\n开始评估...")
    with torch.no_grad():
        for item in tqdm(test_data, desc="评估"):
            prompt = item["prompt"]
            expected = item["completion"]
            
            # 构建消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            predicted = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            
            # 计算指标
            bleu = calculate_bleu(expected, predicted)
            rouge = calculate_rouge_l(expected, predicted)
            
            all_bleu.append(bleu)
            all_rouge.append(rouge)
            
            results.append({
                "prompt": prompt,
                "expected": expected,
                "predicted": predicted,
                "bleu": bleu,
                "rouge_l": rouge,
            })
    
    # 计算平均分数
    avg_bleu = sum(all_bleu) / len(all_bleu)
    avg_rouge = sum(all_rouge) / len(all_rouge)
    
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"平均 BLEU-4: {avg_bleu:.4f}")
    print(f"平均 ROUGE-L: {avg_rouge:.4f}")
    
    # 保存结果
    evaluation_results = {
        "num_samples": len(test_data),
        "avg_bleu": avg_bleu,
        "avg_rouge_l": avg_rouge,
        "samples": results,
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return evaluation_results


def compare_models(
    models: dict,
    test_file: str = "/home/rzzhang/llm_sft/data/test.jsonl",
    max_samples: int = 50,
):
    """比较多个模型"""
    print("="*60)
    print("模型比较")
    print("="*60)
    
    results = {}
    for name, model_path in models.items():
        print(f"\n评估模型: {name}")
        output_file = f"/home/rzzhang/output/eval_{name}.json"
        result = evaluate_model(
            model_path=model_path,
            test_file=test_file,
            output_file=output_file,
            max_samples=max_samples,
        )
        results[name] = {
            "bleu": result["avg_bleu"],
            "rouge": result["avg_rouge_l"],
        }
    
    # 打印比较结果
    print("\n" + "="*60)
    print("模型比较结果")
    print("="*60)
    print(f"{'模型':<20} {'BLEU-4':>10} {'ROUGE-L':>10}")
    print("-" * 42)
    for name, scores in results.items():
        print(f"{name:<20} {scores['bleu']:>10.4f} {scores['rouge']:>10.4f}")
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--model_path", type=str, default="/home/rzzhang/output/sft",
                        help="模型路径")
    parser.add_argument("--test_file", type=str, default="/home/rzzhang/llm_sft/data/test.jsonl",
                        help="测试数据文件")
    parser.add_argument("--output_file", type=str, default="/home/rzzhang/output/evaluation_results.json",
                        help="输出文件")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数")
    parser.add_argument("--compare", action="store_true",
                        help="比较多个模型")
    
    args = parser.parse_args()
    
    if args.compare:
        models = {
            "base": "/home/rzzhang/models/qwen3.5-2b",  # 基座模型
            "sft_rank4": "/home/rzzhang/output/sft_rank4",  # SFT 模型
            #"dpo": "/home/rzzhang/output/dpo",  # DPO 模型
        }
        compare_models(models, args.test_file, args.max_samples or 50)
    else:
        evaluate_model(
            model_path=args.model_path,
            test_file=args.test_file,
            output_file=args.output_file,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()