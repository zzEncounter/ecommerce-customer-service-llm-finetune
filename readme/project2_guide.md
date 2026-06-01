# 项目二：金融研报问答——RAG 系统详细指南

## 项目概述

### 项目目标
给分析师做一套研报问答工具，自然语言提问，系统自动从几百份 PDF 里找到相关段落，生成回答，标注来源文档和页码。

### 为什么选金融场景
1. **PDF 处理难度大**：研报里全是多栏排版、表格、图表、脚注，不做版面分析直接 split，表格数据全是乱的
2. **答案确定性**："比亚迪 Q3 毛利率多少"，研报里白纸黑字写着，评测非常好做
3. **行业需求大**：字节、蚂蚁、百度、阿里都有金融 AI 团队

---

## 第一周：数据处理 + 基线搭建

### 1.1 数据准备

#### 数据来源详细指南

##### 一、金融研报 PDF 数据集

**1. 东方财富网（推荐，数据最全）**
```
网址：https://data.eastmoney.com/notices/stock.html
步骤：
1. 进入"数据中心" → "公告" → "定期报告"
2. 筛选"年报"、"季报"、"研报"
3. 选择目标行业（新能源、半导体、消费等）
4. 下载 PDF 文件

热门公司研报推荐下载：
- 新能源：宁德时代(300750)、比亚迪(002594)、隆基绿能(601012)
- 半导体：中芯国际(688981)、北方华创(002371)、韦尔股份(603501)
- 消费：贵州茅台(600519)、五粮液(000858)、伊利股份(600887)
```

**2. 巨潮资讯网（官方公告平台）**
```
网址：http://www.cninfo.com.cn/
步骤：
1. 选择"上市公司公告"
2. 按"年度报告"、"季度报告"筛选
3. 输入股票代码或公司名称搜索
4. 下载公告 PDF

优点：官方数据源，权威可靠
```

**3. 卷心菜研究（行业研报）**
```
网址：https://www.iresearch.com.cn/
步骤：
1. 进入"研究报告"栏目
2. 选择行业分类（汽车、消费、科技等）
3. 免费注册后下载

优点：行业深度报告，数据详实
```

**4. 发现报告（研报聚合平台）**
```
网址：https://www.fxbaoguo.com/
特点：
- 聚合多个券商研报
- 按行业、主题分类
- 大部分免费下载
```

**5. 萝卜投研（研报搜索）**
```
网址：https://robo.datayes.com/
步骤：
1. 注册登录（免费）
2. 搜索框输入公司名或行业关键词
3. 筛选研报类型下载
```

##### 二、政策文件数据集

**1. 发改委官网**
```
网址：https://www.ndrc.gov.cn/xwdt/tzgg/
内容：
- 产业政策文件
- 发改委公告
- 行业发展规划
```

**2. 人民银行官网**
```
网址：http://www.pbc.gov.cn/
内容：
- 货币政策报告
- 金融统计数据
- 政策解读文件
```

**3. 国务院**
```
网址：http://www.gov.cn/zhengce/
内容：
- 国务院文件
- 政策解读
- 行业发展规划
```

##### 三、评测数据集（用于测试系统效果）

**1. FinanceIQ（金融问答数据集）**
```
GitHub：https://github.com/NiuTrans/FinanceIQ
HuggingFace：https://huggingface.co/datasets/NiuTrans/FinanceIQ
说明：中文金融问答数据集，包含多个金融场景问答
```

**2. FinQA（金融数值推理）**
```
GitHub：https://github.com/czyssrs/FinQA
HuggingFace：https://huggingface.co/datasets/ib-regweek/FinQA
说明：金融数值推理问答，适合测试表格理解能力
```

**3. ConvFinQA（对话式金融问答）**
```
HuggingFace：https://huggingface.co/datasets/yzluka/ConvFinQA
说明：多轮对话金融问答
```

**4. C-FinQA（中文金融问答）**
```
GitHub：https://github.com/xiaoa24/c_finqa
说明：中文金融数值问答数据集
```

##### 四、快速获取脚本示例

```python
# 安装依赖
pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
import os

def download_eastmoney_report(stock_code, save_dir):
    """
    从东方财富下载研报
    示例：download_eastmoney_report("300750", "./reports/")
    """
    # 东方财富公告 API
    url = f"https://data.eastmoney.com/notices/stock/{stock_code}.html"
    
    # 实际使用时需要根据网页结构调整
    # 建议使用 selenium 处理动态页面
    pass

# 批量下载示例
stock_list = [
    "300750",  # 宁德时代
    "002594",  # 比亚迪
    "601012",  # 隆基绿能
    "688981",  # 中芯国际
    "600519",  # 贵州茅台
]

for stock in stock_list:
    print(f"正在下载 {stock} 的研报...")
    # download_eastmoney_report(stock, "./reports/")
```

##### 五、数据量建议

| 数据类型 | 数量建议 | 说明 |
|----------|---------|------|
| 公司年报 | 50-80 份 | 每个行业 15-20 份 |
| 公司季报 | 50-80 份 | 重点是财务表格 |
| 行业研报 | 50-100 份 | 券商研报为主 |
| 政策文件 | 30-50 份 | 发改委、央行文件 |
| **总计** | **200-300 份** | 覆盖 3-5 个行业 |

##### 六、面试数据来源话术

> "语料库构建分三部分：第一部分是上市公司年报季报，从东方财富和巨潮资讯获取，覆盖新能源、半导体、消费三个行业共 200 份；第二部分是行业深度研报，来自卷心菜研究和发现报告平台共 80 份；第三部分是政策文件，来自发改委和人民银行官网共 40 份。所有文档均为 PDF 格式，包含多栏排版和复杂表格。评测集基于 FinanceIQ 并结合业务场景自建了 120 条，分事实型、对比型、汇总型三类问题。"

**数据量控制**：200-300 份 PDF，覆盖 3-5 个行业（如新能源+半导体+消费）

#### 面试话术
> "语料库 300+ 份金融研报和政策文件，覆盖新能源、半导体、消费三个行业。文档以 PDF 为主，包含多栏排版、表格、图表，部分超过 80 页。评测集自建了 120 条，分事实型、对比型、汇总型三类。"

### 1.2 PDF 解析实现

#### 技术方案
```python
# 基础依赖安装
pip install PyMuPDF pdfplumber

# PDF 解析示例代码框架
import fitz  # PyMuPDF
import pdfplumber

def parse_pdf(filepath):
    """
    PDF 解析主函数
    1. 用 PyMuPDF 提取文本
    2. 用 pdfplumber 解析表格
    3. 处理多栏/页眉页脚
    """
    # 文本提取
    doc = fitz.open(filepath)
    text_content = []
    for page in doc:
        text_content.append(page.get_text())
    
    # 表格提取
    tables = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            tables.extend(page.extract_tables())
    
    return text_content, tables
```

#### 面试关键点
能讲清楚"纯文本 split 在 PDF 上为什么不行，表格会被拆散、多栏内容会串行"

### 1.3 切块实验

#### 切块参数对比实验
| 实验组 | chunk_size | overlap | 预期观察指标 |
|--------|-----------|---------|-------------|
| A | 256 | 50 | 切块数量多，上下文可能断裂 |
| B | 512 | 100 | 平衡方案 |
| C | 1024 | 200 | 切块数量少，可能包含噪声 |

#### 金融文档特殊处理
```python
def table_aware_chunking(text, tables, chunk_size=512, overlap=100):
    """
    表格保护切块策略
    - 表格行不能被切断
    - 表格单独切块
    """
    chunks = []
    # 识别表格边界
    # 保护表格完整性
    # 对非表格文本进行切块
    return chunks
```

#### 需要记录的数据
- 每组切块数量
- 平均长度
- 后续召回的 Recall@5 变化

### 1.4 向量索引构建

#### Embedding 模型配置
```python
# 安装 FlagEmbedding
pip install FlagEmbedding

from FlagEmbedding import FlagModel
model = FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")

# 生成向量
embeddings = model.encode_queries(queries)
```

#### FAISS 索引构建
```python
import faiss

# Flat 索引（精确搜索，用于 baseline）
dimension = 1024  # bge-large-zh-v1.5 的维度
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# 保存索引
faiss.write_index(index, "faiss_index.bin")
```

#### 输出要求
- 索引构建脚本
- 向量维度记录
- 文档数量记录

---

## 第二周：多路召回 + 召回评测

### 2.1 三路召回实现

#### 路线 A：纯向量召回
```python
def vector_search(query_embedding, index, top_k=10):
    """
    纯向量召回
    """
    D, I = index.search(query_embedding, top_k)
    return D, I  # 距离和索引
```

#### 路线 B：纯 BM25 召回
```python
# 安装 BM25
pip install rank_bm25

from rank_bm25 import BM25Okapi

def bm25_search(query, tokenized_corpus, top_k=10):
    """
    纯 BM25 召回
    """
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]
```

#### 路线 C：混合召回
```python
def hybrid_search(query, query_embedding, vector_index, bm25, 
                  alpha=0.5, top_k=10):
    """
    混合召回 = alpha * 向量分数 + (1-alpha) * BM25分数
    """
    # 向量召回
    vector_scores, vector_indices = vector_search(query_embedding, vector_index, top_k*2)
    
    # BM25 召回
    bm25_indices, bm25_scores = bm25_search(query, bm25, top_k*2)
    
    # 分数融合
    combined_scores = {}
    for i, idx in enumerate(vector_indices):
        combined_scores[idx] = alpha * normalize(vector_scores[i])
    for i, idx in enumerate(bm25_indices):
        if idx in combined_scores:
            combined_scores[idx] += (1-alpha) * normalize(bm25_scores[i])
        else:
            combined_scores[idx] = (1-alpha) * normalize(bm25_scores[i])
    
    # 排序返回
    sorted_results = sorted(combined_scores.items(), 
                           key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_results
```

### 2.2 评测集构建

#### 评测数据格式
```json
{
    "query": "比亚迪 Q3 营收多少",
    "ground_truth_doc_id": "byd_2024q3_report.pdf",
    "ground_truth_page": 15,
    "answer": "比亚迪 2024 年 Q3 营收为 1621.8 亿元",
    "query_type": "fact"
}
```

#### 三类查询类型
| 类型 | 示例 | 特点 |
|------|------|------|
| 事实型 | "比亚迪 Q3 营收多少" | 单一答案，确定性高 |
| 对比型 | "宁德时代 vs 比亚迪毛利率哪个高" | 需要跨文档对比 |
| 汇总型 | "新能源行业 2025 年主要政策变化" | 需要汇总多文档信息 |

**数量建议**：每类 40 条，共 120 条评测集

### 2.3 评测指标计算

```python
def calculate_recall_at_k(retrieved_docs, ground_truth, k):
    """
    计算 Recall@K
    """
    retrieved_set = set(retrieved_docs[:k])
    ground_truth_set = set(ground_truth)
    
    if len(ground_truth_set) == 0:
        return 0.0
    
    intersection = len(retrieved_set & ground_truth_set)
    return intersection / len(ground_truth_set)
```

### 2.4 召回方案对比实验

#### 必须完成的实验表格
| 召回方案 | Recall@3 | Recall@5 | Recall@10 | 备注 |
|----------|----------|----------|-----------|------|
| 纯向量 | ? | ? | ? | baseline |
| 纯 BM25 | ? | ? | ? | 关键词场景更强 |
| 混合召回 | ? | ? | ? | 通常最优 |

### 2.5 FAISS 索引类型对比

#### 三种索引类型实验
```python
# 1. Flat（精确搜索）
index_flat = faiss.IndexFlatIP(dimension)

# 2. IVF（倒排文件索引）
nlist = 100  # 聚类中心数量
quantizer = faiss.IndexFlatIP(dimension)
index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# 3. HNSW（层次导航小世界图）
M = 32  # 每个节点的连接数
index_hnsw = faiss.IndexHNSWFlat(dimension, M)
```

#### 对比指标
| 索引类型 | Recall@10 | 查询延迟(ms) | 构建时间(s) | 内存占用(MB) |
|----------|-----------|-------------|------------|-------------|
| Flat | ? | ? | ? | ? |
| IVF | ? | ? | ? | ? |
| HNSW | ? | ? | ? | ? |

---

## 第三周：重排 + 生成 + 故障处理

### 3.1 Reranker 接入

#### 安装与配置
```python
pip install FlagEmbedding

from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

def rerank(query, candidates, top_k=5):
    """
    对召回结果进行重排
    """
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.compute_score(pairs)
    
    # 按分数排序
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [candidates[i] for i in ranked_indices]
```

#### 重排对比实验
| 方案 | Recall@5 | 准确率 | 延迟增加 |
|------|----------|--------|----------|
| 向量召回 Top5 | ? | ? | baseline |
| 向量召回 Top20 → Rerank Top5 | ? | ? | +? ms |

### 3.2 生成模块

#### 引用溯源实现
```python
def generate_with_citation(query, retrieved_docs, model):
    """
    生成带引用的回答
    """
    prompt = f"""
    基于以下文档内容回答问题，并在回答中标注引用来源。
    
    文档内容：
    {format_docs_with_source(retrieved_docs)}
    
    问题：{query}
    
    回答格式要求：
    1. 回答内容
    2. 引用来源 [文档名, 页码]
    """
    
    response = model.generate(prompt)
    return response
```

#### 拒答机制
```python
def answer_with_rejection(query, retrieved_docs, score_threshold=0.3):
    """
    低置信度时拒答
    """
    max_score = max([doc['score'] for doc in retrieved_docs])
    
    if max_score < score_threshold:
        return {
            "answer": "抱歉，我没有找到相关信息，无法确定回答。",
            "citations": [],
            "confidence": max_score
        }
    
    return generate_with_citation(query, retrieved_docs)
```

### 3.3 故障处理案例

#### 典型故障及解决方案
| 故障 | 示例 | 根因 | 解决方案 |
|------|------|------|----------|
| 空召回 | "宁德时代 2025 年装机量"，文档写的是"装车量" | 表述差异导致 embedding 语义鸿沟 | query 改写（同义词扩展） |
| 答非所问 | 问"比亚迪毛利率"召回了营收段落 | chunk 太大 | 缩小 chunk + 表格单独切块 + 重排 |
| 幻觉 | 模型编造了不存在的数字 | 生成时未绑定证据 | 引用约束 + 后校验 |
| 延迟高 | 响应超过 3 秒 | 重排模型太重 | 先粗筛再精排 / 缓存热查询 |

#### Query 改写示例
```python
def query_rewrite(query):
    """
    同义词扩展改写
    """
    synonym_dict = {
        "装机量": ["装车量", "部署量"],
        "营收": ["营业收入", "销售额"],
        "毛利率": ["毛利润率", " gross margin"]
    }
    
    expanded_queries = [query]
    for term, synonyms in synonym_dict.items():
        if term in query:
            for syn in synonyms:
                expanded_queries.append(query.replace(term, syn))
    
    return expanded_queries
```

---

## 第四周：评测闭环 + Demo + 文档

### 4.1 Demo 搭建

#### Gradio 界面示例
```python
import gradio as gr

def rag_query(query):
    """
    RAG 查询主函数
    """
    # 1. 召回
    retrieved_docs = hybrid_search(query, ...)
    
    # 2. 重排
    reranked_docs = rerank(query, retrieved_docs)
    
    # 3. 生成
    answer = generate_with_citation(query, reranked_docs)
    
    return answer

demo = gr.Interface(
    fn=rag_query,
    inputs=gr.Textbox(label="输入您的问题"),
    outputs=[
        gr.Textbox(label="回答"),
        gr.JSON(label="引用来源")
    ],
    title="金融研报问答系统"
)

demo.launch()
```

### 4.2 项目文档结构

```
project2_rag/
├── data/
│   ├── raw_pdfs/           # 原始 PDF
│   ├── parsed/             # 解析后文本
│   └── eval_set.json       # 评测集
├── scripts/
│   ├── pdf_parser.py       # PDF 解析
│   ├── chunking.py         # 切块
│   ├── indexing.py         # 索引构建
│   └── evaluate.py         # 评测脚本
├── models/
│   ├── embedding/          # Embedding 模型
│   └── reranker/           # Reranker 模型
├── experiments/
│   ├── chunk_exp.md        # 切块实验记录
│   ├── recall_exp.md       # 召回实验记录
│   └── rerank_exp.md       # 重排实验记录
├── demo/
│   └── app.py              # Gradio Demo
├── docs/
│   ├── architecture.md     # 系统架构图
│   ├── badcase.md          # Badcase 池
│   └── summary.md          # 项目总结
└── README.md
```

### 4.3 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户查询                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Query 改写/扩展                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      混合召回                                │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │   向量召回     │  │   BM25 召回   │  │   分数融合     │    │
│  │   (FAISS)     │  │  (rank_bm25)  │  │   (加权)       │    │
│  └───────────────┘  └───────────────┘  └───────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Reranker 重排                             │
│                (bge-reranker-v2-m3)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    生成模块                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Prompt 构建 → LLM 生成 → 引用绑定 → 拒答判断         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              返回结果 (回答 + 引用来源)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 技术选型总览

| 组件 | 推荐 | 备选 | 说明 |
|------|------|------|------|
| 向量检索 | FAISS | Milvus | 数据量大时用 Milvus |
| Embedding | bge-large-zh-v1.5 | text2vec, m3e | FlagEmbedding 出品 |
| 重排 | bge-reranker-v2-m3 | ms-marco | Cross-Encoder |
| BM25 | rank_bm25 | Elasticsearch | Python 库即可 |
| 生成模型 | Qwen3-8B | DeepSeek-R1-Distill-7B | |
| 评测 | RAGAS / 自建 | | |
| PDF 解析 | PyMuPDF + pdfplumber | marker, RAGFlow DeepDoc | |
| 快速原型 | LlamaIndex | LangChain | 仅搭 baseline |

---

## 面试高频问题与回答

### Q1: chunk size 怎么选的？
> 讲你的 256/512/1024 三组对比，最好记住具体 Recall 数字。在我的实验中，chunk_size=512 达到了 Recall 和上下文完整性的最佳平衡。

### Q2: 为什么不只用向量检索？
> 举例子：用户搜"宁德时代装机量"，文档里写的是"装车量"，纯向量可能召不回来，BM25 反而能匹配到。所以混合召回效果最稳。

### Q3: Reranker 到底有多大提升？
> 直接报数。在我的实验中，加入 Reranker 后 Recall@5 从 X% 提升到 Y%，提升约 Z 个百分点。

### Q4: 空召回你怎么处理？
> query 改写 + 同义词扩展 + 降级拒答。具体来说，我会先对 query 进行同义词扩展，如果还是空召回，则返回"抱歉未找到相关信息"。

### Q5: 幻觉怎么缓解？
> 证据绑定 + 低置信度拒答 + 生成后引用校验。具体：
> 1. 回答必须绑定召回文档作为证据
> 2. 召回分数低于阈值时拒答
> 3. 生成后校验引用的数字是否在原文中存在

### Q6: QPS 多少，瓶颈在哪？
> 重排模型和生成模型是延迟大头。在我的压测中，QPS 约为 X，P95 延迟 Y ms，瓶颈主要在 Reranker 和 LLM 生成阶段。

---

## 简历写法

> 面向金融研报智能问答场景，针对 PDF 多栏排版、表格密集等复杂文档特点，设计并实现 RAG 系统。构建 BM25+Dense 混合召回链路，引入 Cross-Encoder 重排提升检索相关性；设计"表格保护"切块策略解决财务表格切断问题，并基于自建 120 条三类型评测集和 badcase 池迭代优化。最终混合召回 Recall@5 达 87%，较纯向量方案提升 15 pp，答非所问比例下降 22%。

---

## 开源仓库参考

| 仓库 | Stars | 重点学习内容 |
|------|-------|-------------|
| [RAGFlow](https://github.com/infiniflow/ragflow) | 75k | DeepDoc PDF 解析 + 检索融合 + 评测体系 |
| [FAISS](https://github.com/facebookresearch/faiss) | - | 向量检索 |
| [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | - | Embedding + Reranker |
| [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) | 37k | 中文 RAG 经典参考，切块/向量库适配/混合检索 |
| [QAnything](https://github.com/netease-youdao/QAnything) | 14k | 2-stage retrieval + rerank 阈值设计 |
| [LlamaIndex](https://github.com/run-llama/llama_index) | - | 快速搭 baseline |
| [Milvus](https://github.com/milvus-io/milvus) | - | 生产级向量库 |
| [GraphRAG](https://github.com/microsoft/graphrag) | - | 进阶：复杂多跳问答场景 |

---

## 学习路径建议

### 第一阶段：代码阅读（2-3 天）
1. 读 RAGFlow 的 DeepDoc 模块，理解 PDF 解析思路
2. 读 Langchain-Chatchat 的混合检索实现
3. 读 QAnything 的 2-stage retrieval 策略

### 第二阶段：实现（3-4 周）
按照上述四周计划执行，每周完成对应任务

### 第三阶段：面试准备
1. 整理所有实验数据表格
2. 准备 badcase 分析文档
3. 能够流畅回答上述面试问题
4. 准备 Demo 演示

---

## 关键检查点

每完成一个阶段，用以下标准检查：

- [ ] 有没有至少 3 组对比实验
- [ ] 有没有能说出口的数字（"效果不错"是废话，"Recall@5 从 72% 到 87%"才是有效信息）
- [ ] 有没有翻车记录和解决方案
- [ ] 能不能讲满 5 分钟不卡壳（背景→问题→方案→实验→结果→复盘）
- [ ] 面试官追问三层还能不能接住

五条里不到三条的项目，建议继续完善后再写进简历。