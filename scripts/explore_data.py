# -*- coding: utf-8 -*-
"""探索数据集结构和格式"""
import os

def explore_data_file(filepath, num_lines=10):
    """探索单个数据文件"""
    print(f"\n{'='*60}")
    print(f"文件: {filepath}")
    print(f"{'='*60}")
    
    # 统计总行数
    total_lines = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"总行数: {total_lines}")
    
    # 读取前几行
    print(f"\n前 {num_lines} 行示例:")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i in range(num_lines):
            line = next(f).strip()
            parts = line.split('\t')
            print(f"\n--- 行 {i+1} ---")
            print(f"字段数量: {len(parts)}")
            print(f"标签 (label): {parts[0]}")
            if len(parts) > 1:
                # 倒数第二部分是对话历史，最后一部分是回复
                conversation = parts[1:-1] if len(parts) > 2 else []
                response = parts[-1] if len(parts) > 1 else ""
                print(f"对话轮数: {len(conversation)}")
                print(f"对话内容: {' | '.join(conversation[:3])}{'...' if len(conversation) > 3 else ''}")
                print(f"回复: {response[:50]}{'...' if len(response) > 50 else ''}")

def main():
    data_dir = "llm/E-commerce dataset/E-commerce dataset"
    
    # 探索各个文件
    for filename in ["train.txt", "dev.txt", "test.txt"]:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            explore_data_file(filepath, num_lines=5)

if __name__ == "__main__":
    main()