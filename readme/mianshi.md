# 大模型微调项目面试文档

## 一、项目介绍（面试版本）

### 1.1 项目概述
这是一个基于 Qwen2.5-3B 大语言模型的电商客服对话系统微调项目。项目实现了完整的训练流程，包括：
- **SFT（监督微调）**：使用电商客服对话数据进行监督学习
- **DPO（直接偏好优化）**：使用正负样本进行偏好对齐
- **LoRA 高效微调**：显著减少训练参数量

### 1.2 技术栈
- **基座模型**：Qwen2.5-3B（阿里通义千问）
- **微调方法**：LoRA（Low-Rank Adaptation）
- **训练框架**：Transformers + PEFT + TRL
- **评估指标**：BLEU、ROUGE-L
- **精度**：BFloat16 混合精度训练

### 1.3 项目成果
| 模型 | BLEU | ROUGE-L | 说明 |
|------|------|---------|------|
| Base Model | 0.134 | 0.398 | 基座模型 |
| SFT (rank=8) | 0.202 | 0.454 | SFT微调后 |
| SFT (rank=4) | - | - | 低秩对比实验 |
| DPO | - | - | 偏好对齐后 |

---

## 二、大模型微调核心知识点

### 2.1 LoRA（Low-Rank Adaptation）

#### 原理
LoRA 的核心思想是：模型微调时的权重更新矩阵具有低秩特性。

原始权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，微调后的权重为：
$$W = W_0 + \Delta W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d, k)$。

#### 参数量对比
- 原始参数量：$d \times k$
- LoRA 参数量：$r \times (d + k)$
- 压缩比：$\frac{r(d+k)}{dk} \approx \frac{2r}{d}$（当 $d \approx k$）

#### 本项目配置
```python
LoraConfig(
    r=8,                    # 秩
    lora_alpha=32,          # 缩放系数
    lora_dropout=0.1,       # dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
)
```

#### Alpha 的作用
实际更新量为：$\frac{\alpha}{r} \cdot BA$
- $\alpha$ 控制更新幅度
- 本项目 $\alpha/r = 32/8 = 4$，放大了低秩更新的影响

---

### 2.2 SFT（Supervised Fine-Tuning）

#### 训练目标
最大化模型在目标数据上的对数似然：
$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}}[\log \pi_\theta(y|x)]$$

#### 数据处理
本项目将电商对话数据转换为指令格式：
```
用户: 你好
客服: 您好，请问有什么可以帮您？
用户: 想问一下发货时间
客服: ...
```

#### 关键训练参数
```python
TrainingArguments(
    learning_rate=2e-4,        # 学习率
    num_train_epochs=3,        # 训练轮数
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    bf16=True,                 # 混合精度
    gradient_checkpointing=True,  # 梯度检查点节省显存
)
```

---

### 2.3 DPO（Direct Preference Optimization）

#### 背景
传统 RLHF 需要训练奖励模型和使用 PPO 优化，流程复杂且不稳定。DPO 直接优化偏好，无需显式奖励模型。

#### 核心原理
DPO 的损失函数：
$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

其中：
- $y_w$：chosen（优选回复）
- $y_l$：rejected（劣选回复）
- $\pi_{ref}$：参考模型（SFT 后的模型）
- $\beta$：温度参数，控制偏好强度

#### 本项目实现
```python
# 构建偏好对
preference_pairs = [{
    "prompt": "用户: ...",
    "chosen": "好的亲，这边给您处理...",    # 正样本回复
    "rejected": "不知道，你自己看...",      # 负样本回复
}]

# DPO 训练
DPOConfig(
    beta=0.1,           # 温度参数
    learning_rate=5e-6, # 较小的学习率
)
```

#### DPO vs RLHF
| 方面 | RLHF | DPO |
|------|------|-----|
| 复杂度 | 高（需要奖励模型+PPO） | 低（直接优化） |
| 稳定性 | 不稳定 | 稳定 |
| 计算成本 | 高 | 低 |
| 效果 | 可调优空间大 | 依赖数据质量 |

---

### 2.4 评估指标

#### BLEU
主要评估 n-gram 重合度，常用于机器翻译：
$$BLEU = BP \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)$$

- $p_n$：n-gram 精确度
- $BP$：简短惩罚因子

#### ROUGE-L
基于最长公共子序列（LCS）：
$$R_{LCS} = \frac{LCS(X,Y)}{m}$$
$$P_{LCS} = \frac{LCS(X,Y)}{n}$$
$$F_{LCS} = \frac{(1+\beta^2)R_{LCS}P_{LCS}}{R_{LCS} + \beta^2 P_{LCS}}$$

#### 本项目评估结果分析
- BLEU=0.202：说明词汇匹配度中等
- ROUGE-L=0.454：说明句式结构学习较好
- 客服对话任务中，语义正确性比精确匹配更重要

---

### 2.5 训练优化技术

#### 混合精度训练（BF16）
- 使用 BFloat16 格式，动态范围与 Float32 相同
- 减少显存占用约 50%
- 加速训练约 2x

#### 梯度累积
```python
gradient_accumulation_steps=4
# 实际 batch_size = per_device_batch_size × gradient_accumulation_steps
# = 4 × 4 = 16
```

#### 梯度检查点
```python
gradient_checkpointing=True
# 牺牲 20% 计算速度，换取 50%+ 显存节省
```

---

## 三、面试常见问题与回答

### Q1：为什么选择 LoRA 而不是全量微调？

**回答要点：**
1. **参数效率**：LoRA 只训练约 0.1%-1% 的参数。以本项目为例：
   - 3B 模型全量微调需要约 6GB 显存（仅参数）
   - LoRA 只需约 10-50MB 可训练参数

2. **显存优势**：全量微调 3B 模型需要约 24GB+ 显存，LoRA 可在 8GB 显存上完成

3. **避免灾难性遗忘**：LoRA 保持原权重不变，只学习增量更新

4. **模型切换**：可保存多个 LoRA 适配器，动态切换不同任务

5. **实际项目考虑**：电商客服场景数据量有限（约数万条），全量微调容易过拟合

---

### Q2：LoRA 的 rank 应该怎么选择？

**回答要点：**
1. **经验值**：通常 4-64 之间，本项目使用 8

2. **秩越大**：
   - 表达能力越强，可学习更复杂的模式
   - 但参数量增加，可能过拟合
   - 本项目做了 rank=4/8/16 的对比实验

3. **秩越小**：
   - 参数效率高，泛化可能更好
   - 但表达能力受限

4. **选择依据**：
   - 任务复杂度：客服对话相对简单，rank=8 足够
   - 数据量：数据量大可适当增大 rank
   - 显存限制：rank 小则显存占用少

5. **目标模块选择**：
   - 通常选择 attention 层（q、k、v、o_proj）
   - 本项目额外包括 MLP 层（gate、up、down_proj）
   - 更全面的覆盖带来更好的效果

---

### Q3：DPO 的 beta 参数如何理解？

**回答要点：**
1. **beta 的作用**：控制模型对偏好的敏感程度
   - $\beta$ 大：模型对 chosen 和 rejected 的差异更敏感
   - $\beta$ 小：模型更新更保守

2. **本项目设置**：$\beta=0.1$（较小值）
   - 原因：客服对话的正确性差异不是极端的
   - "好"和"差"的回复在语义上可能部分重叠

3. **调参经验**：
   - 偏好差异明显（如有害 vs 无害）：$\beta \in [0.1, 0.5]$
   - 偏好差异微妙（如风格差异）：$\beta \in [0.05, 0.1]$

4. **与学习率的配合**：
   - DPO 学习率通常比 SFT 小一个数量级
   - 本项目 SFT lr=2e-4，DPO lr=5e-6

---

### Q4：SFT 和 DPO 的训练顺序是什么？为什么？

**回答要点：**
1. **正确顺序**：Pre-training → SFT → DPO（或 RLHF）

2. **原因**：
   - **SFT 先行**：让模型学会基本的对话能力和任务格式
   - **DPO 后续**：在已有能力基础上进行偏好对齐，微调回复风格

3. **如果顺序反了**：
   - DPO 数据通常是格式化的偏好对，模型需要先理解格式
   - 未经 SFT 的模型可能无法正确理解 chosen/rejected 的含义

4. **本项目实现**：
   ```python
   # 先加载 SFT 模型作为 DPO 的起点
   model = PeftModel.from_pretrained(model, sft_model_path)
   model = model.merge_and_unload()  # 合并 LoRA 权重
   # 然后创建新的 LoRA 进行 DPO 训练
   model = get_peft_model(model, new_lora_config)
   ```

---

### Q5：如何构建 DPO 的偏好对数据？

**回答要点：**
1. **数据来源**：
   - 人工标注（质量最高，成本高）
   - 模型生成+评分（如使用 GPT-4 打分）
   - 现有数据挖掘（本项目方法）

2. **本项目方案**：
   - 原始数据已标注 label（1=好回复，0=差回复）
   - 相同对话上下文的正负样本配对
   - ```python
     # 构建偏好对
     for context, responses in data_by_context.items():
         for chosen in responses["chosen"]:
             if responses["rejected"]:
                 preference_pairs.append({
                     "prompt": context,
                     "chosen": chosen,
                     "rejected": responses["rejected"][0]
                 })
     ```

3. **数据质量要求**：
   - chosen 必须明显优于 rejected
   - 两者应该针对相同的上下文/问题
   - 避免模糊的偏好差异

4. **数据量**：
   - DPO 不需要大量数据
   - 本项目限制为 10000 对
   - 质量比数量更重要

---

### Q6：训练中遇到显存不足怎么办？

**回答要点：**
1. **LoRA 配置优化**：
   - 减小 rank
   - 减少 target_modules（只保留 attention 层）

2. **训练参数调整**：
   - 减小 per_device_train_batch_size
   - 增大 gradient_accumulation_steps（保持等效 batch size）
   - 启用 gradient_checkpointing

3. **精度优化**：
   - 使用 bf16 或 fp16 混合精度
   - 考虑 8bit 量化（QLoRA）

4. **本项目配置**：
   ```python
   per_device_train_batch_size=4      # 较小 batch
   gradient_accumulation_steps=4      # 累积梯度
   gradient_checkpointing=True        # 节省显存
   bf16=True                          # 混合精度
   ```

5. **估算公式**：
   - 显存 ≈ 模型参数 × 2（fp16） + 梯度 × 2 + 优化器状态 × 8
   - LoRA 大幅减少后两项

---

### Q7：如何评估微调效果？

**回答要点：**
1. **自动评估指标**：
   - BLEU：词汇精确匹配
   - ROUGE-L：最长公共子序列
   - Perplexity：困惑度（越低越好）

2. **本项目评估方案**：
   - 保留 50 条测试样本
   - 计算 BLEU 和 ROUGE-L
   - 对比基座模型 vs SFT vs DPO

3. **局限性**：
   - BLEU/ROUGE 关注表面匹配
   - 客服对话中语义正确性更重要
   - 例："好的亲" vs "好的呢亲" BLEU 较低但语义相同

4. **人工评估维度**：
   - 回复相关性
   - 服务态度（礼貌性）
   - 信息准确性
   - 解决问题能力

5. **实际生产中的评估**：
   - A/B 测试
   - 用户满意度调查
   - 业务指标（如问题解决率）

---

### Q8：项目中的难点和解决方案？

**回答要点：**
1. **数据格式处理**：
   - 原始数据为 tab-separated 格式
   - 需要转换为模型可理解的对话格式
   - 解决：编写数据预处理脚本，统一格式

2. **DPO 数据构建**：
   - 难点：如何构建高质量的偏好对
   - 解决：利用已有正负标注，按对话上下文配对

3. **模型加载与合并**：
   - DPO 需要先加载 SFT 权重
   - 解决：使用 merge_and_unload() 合并 LoRA 到基座模型

4. **显存管理**：
   - 3B 模型训练显存紧张
   - 解决：梯度检查点 + 混合精度 + LoRA

5. **评估指标局限性**：
   - 自动指标不能完全反映对话质量
   - 解决：结合人工评估，关注实际效果

---

### Q9：如果让你改进这个项目，你会怎么做？

**回答要点：**
1. **数据层面**：
   - 数据增强：同义改写、回译
   - 引入更多客服场景数据
   - 优化偏好对质量（模型打分筛选）

2. **模型层面**：
   - 尝试更大的基座模型（如 7B）
   - 实验其他微调方法（AdaLoRA、QLoRA）
   - 模型集成

3. **训练层面**：
   - 更丰富的超参搜索
   - 多阶段训练（SFT → DPO → SFT 循环）
   - 加入 RLHF 对比实验

4. **评估层面**：
   - 引入基于 LLM 的评估（如 GPT-4 打分）
   - 构建客服领域专用测试集
   - 人工评估流程标准化

5. **部署层面**：
   - 模型量化（INT8/INT4）
   - 推理优化（vLLM、TensorRT）
   - 构建 API 服务

---

### Q10：请介绍 Qwen 模型的特点

**回答要点：**
1. **模型架构**：
   - 基于 Transformer 的 Decoder-only 架构
   - RoPE 旋转位置编码
   - SwiGLU 激活函数
   - RMSNorm 归一化

2. **Qwen2.5 特性**：
   - 支持长上下文（最长 128K）
   - 多语言支持优秀
   - 中文能力突出
   - 支持 Function Calling

3. **适合本项目的原因**：
   - 3B 版本适合资源有限场景
   - 中文客服对话能力强
   - 开源且社区活跃

4. **与其他模型对比**：
   - vs LLaMA：Qwen 中文能力更强
   - vs ChatGLM：Qwen 生态更完善
   - vs Baichuan：Qwen 性能更均衡

---

## 四、项目技术细节补充

### 4.1 数据处理流程
```
原始数据 (train.txt)
    ↓
解析 tab 分隔格式
    ↓
构建对话格式
    ↓
转换为 JSONL
    ↓
Tokenization
    ↓
训练数据集
```

### 4.2 模型保存与加载
```python
# 保存 LoRA 适配器
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 加载 LoRA 模型
base_model = AutoModelForCausalLM.from_pretrained(base_path)
model = PeftModel.from_pretrained(base_model, lora_path)

# 合并 LoRA 权重（用于推理加速）
model = model.merge_and_unload()
```

### 4.3 推理优化
```python
# 启用 KV Cache 加速推理
model.config.use_cache = True

# 生成参数
generation_config = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}
```

---

## 五、面试技巧总结

1. **回答结构**：先说结论，再展开细节，最后举例说明
2. **结合项目**：每个问题都尽量联系本项目的实际实现
3. **承认局限**：主动说明项目的不足和改进方向
4. **展示深度**：不只是使用框架，要理解底层原理
5. **准备追问**：每个回答都可能引发追问，做好准备

---

*文档版本：v1.0*
*更新时间：2026年5月*