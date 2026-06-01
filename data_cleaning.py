# -*- coding: utf-8 -*-
"""
电商客服数据清洗脚本
按照 project1_guide.md 的要求进行数据清洗
"""
import json
import os
from collections import Counter
from tqdm import tqdm

def clean_ecommerce_data(input_file, output_file, min_response_len=20):
    """
    清洗电商客服数据
    
    数据格式: label \t conversation_utterances \t response
    - label: 1 表示正确回复，0 表示错误回复
    - conversation: 多轮对话，用 \t 分隔
    - response: 客服回复
    
    清洗步骤:
    1. 只保留 label=1 的正样本
    2. 去重（相同的对话上下文）
    3. 去短回复（<20字）
    4. 去矛盾答案（包含"不知道"或"不清楚"）
    5. 格式化为训练格式
    """
    
    print(f"处理文件: {input_file}")
    
    # 统计信息
    stats = {
        "原始数量": 0,
        "正样本数量": 0,
        "去重数量": 0,
        "短回复数量": 0,
        "无效回复数量": 0,
        "最终数量": 0
    }
    
    cleaned_data = []
    seen_contexts = set()  # 去重用的对话上下文集合
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="清洗数据"):
            stats["原始数量"] += 1
            
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            label = parts[0]
            
            # 1. 只保留 label=1 的正样本
            if label != '1':
                continue
            stats["正样本数量"] += 1
            
            # 对话内容（去掉第一个label和最后一个response）
            conversation_parts = parts[1:-1]
            response = parts[-1]
            
            # 构建对话上下文（用于去重）
            context = '\t'.join(conversation_parts)
            
            # 2. 去重
            if context in seen_contexts:
                stats["去重数量"] += 1
                continue
            seen_contexts.add(context)
            
            # 3. 去短回复（<20字）
            if len(response) < min_response_len:
                stats["短回复数量"] += 1
                continue
            
            # 4. 去矛盾答案
            if "不知道" in response or "不清楚" in response or "无法回答" in response:
                stats["无效回复数量"] += 1
                continue
            
            # 构建对话格式
            # 将多轮对话合并为一个字符串
            conversation_text = ""
            for i, utterance in enumerate(conversation_parts):
                if i % 2 == 0:
                    # 用户发言
                    conversation_text += f"用户: {utterance}\n"
                else:
                    # 客服发言
                    conversation_text += f"客服: {utterance}\n"
            
            # 5. 格式化为训练格式
            cleaned_data.append({
                "conversation": conversation_text.strip(),
                "response": response,
                # SFT训练格式
                "instruction": conversation_text.strip(),
                "output": response
            })
            stats["最终数量"] += 1
    
    # 保存清洗后数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n清洗统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return cleaned_data, stats


def convert_to_sft_format(input_file, output_file):
    """
    转换为 TRL SFT 训练格式
    格式: ### 指令:\n{prompt}\n\n### 回答:\n{response}
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        prompt = item['instruction']
        response = item['output']
        
        # TRL SFT 格式：一个 text 字段包含完整对话
        formatted_data.append({
            "prompt": prompt,
            "completion": response,
            "text": f"### 指令:\n{prompt}\n\n### 回答:\n{response}"
        })
    
    # 保存为 JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成: {len(formatted_data)} 条数据 -> {output_file}")
    return formatted_data


def main():
    """主函数：执行数据清洗流程"""
    
    # 数据路径
    data_dir = "llm/E-commerce dataset/E-commerce dataset"
    output_dir = "llm/data"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cleaned"), exist_ok=True)
    
    print("="*60)
    print("电商客服数据清洗")
    print("="*60)
    
    # 1. 清洗训练数据
    print("\n[1/3] 清洗训练数据...")
    train_data, train_stats = clean_ecommerce_data(
        os.path.join(data_dir, "train.txt"),
        os.path.join(output_dir, "cleaned", "train_cleaned.json")
    )
    
    # 2. 清洗验证数据
    print("\n[2/3] 清洗验证数据...")
    dev_data, dev_stats = clean_ecommerce_data(
        os.path.join(data_dir, "dev.txt"),
        os.path.join(output_dir, "cleaned", "dev_cleaned.json")
    )
    
    # 3. 清洗测试数据
    print("\n[3/3] 清洗测试数据...")
    test_data, test_stats = clean_ecommerce_data(
        os.path.join(data_dir, "test.txt"),
        os.path.join(output_dir, "cleaned", "test_cleaned.json")
    )
    
    # 4. 转换为 SFT 训练格式
    print("\n[4/4] 转换为 SFT 训练格式...")
    convert_to_sft_format(
        os.path.join(output_dir, "cleaned", "train_cleaned.json"),
        os.path.join(output_dir, "train.jsonl")
    )
    convert_to_sft_format(
        os.path.join(output_dir, "cleaned", "dev_cleaned.json"),
        os.path.join(output_dir, "val.jsonl")
    )
    convert_to_sft_format(
        os.path.join(output_dir, "cleaned", "test_cleaned.json"),
        os.path.join(output_dir, "test.jsonl")
    )
    
    # 汇总统计
    print("\n" + "="*60)
    print("数据清洗完成!")
    print("="*60)
    print(f"训练集: {train_stats['最终数量']} 条")
    print(f"验证集: {dev_stats['最终数量']} 条")
    print(f"测试集: {test_stats['最终数量']} 条")
    print(f"\n数据保存在: {output_dir}")


if __name__ == "__main__":
    main()