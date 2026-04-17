"""
训练数据准备脚本

为 MLP Router 准备训练数据

训练数据格式:
[
    {
        "question": "What is X?",
        "ground_truth": "X is a...",
        "retrieved_snippets": {
            "sentence": ["retrieved chunk 1", "retrieved chunk 2"],
            "paragraph": ["retrieved chunk 1", "retrieved chunk 2"],
            "section": ["retrieved chunk 1", "retrieved chunk 2"],
            "document": ["retrieved chunk 1"]
        },
        "soft_labels": [0.8, 0.2, 0.0, 0.0]
    },
    ...
]

Soft Labels 构建原理:
1. 对每个问题，用 TF-IDF 在各粒度的 chunks 中快速检索候选片段
2. 用 stsb-roberta-large 计算检索结果与 ground truth 的语义相似度
3. 相似度最高的粒度权重最高

使用示例:
    python src/chunking/prepare_training_data.py --qa-pairs evaluation_set_100papers.json
"""

import os
import json
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chunking.unified_format import MinerUToUnifiedConverter
from src.chunking.granularity_chunker import GranularityChunker, ChunkGranularity
from src.chunking.mlp_router import SoftLabelBuilder, RetrievalResult


def extract_arxiv_id(paper_id: str) -> str:
    """
    从 URL 或 ID 中提取 arXiv ID
    
    Examples:
        "https://arxiv.org/abs/2411.01123" → "2411.01123"
        "2411.01123" → "2411.01123"
    """
    # 尝试从 URL 中提取
    match = re.search(r'(\d{4}\.\d{4,5})', paper_id)
    if match:
        return match.group(1)
    return paper_id


class ChunkRetriever:
    """
    多粒度 Chunk 检索器
    
    对每个问题，在不同粒度的 chunks 中检索相关片段
    """
    
    def __init__(self):
        self.tfidf_vectorizers = {}  # 每个粒度一个 vectorizer
        self.chunks_by_granularity = {}  # {granularity: [(id, text), ...]}
        self.paper_vectorizers = {}  # {granularity: {paper_id: {...}}}
    
    def build_index(self, chunks: List[Dict], granularity: str):
        """
        为指定粒度的 chunks 构建 TF-IDF 索引
        
        Args:
            chunks: chunk 列表 [{"id": "...", "text": "..."}, ...]
            granularity: 粒度名称
        """
        texts = [c.get('text', '') for c in chunks]
        
        # 过滤空文本
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        valid_chunks = [chunks[i] for i in valid_indices]
        
        if not valid_texts:
            return
        
        # 训练 TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        
        self.tfidf_vectorizers[granularity] = {
            'vectorizer': vectorizer,
            'matrix': tfidf_matrix,
            'chunks': valid_chunks
        }
        self.chunks_by_granularity[granularity] = valid_chunks

        # 同时构建按 paper_id 划分的索引，保证 QA 仅与所属论文对齐
        paper_chunks: Dict[str, List[Dict]] = {}
        for chunk in valid_chunks:
            paper_id = chunk.get('paper_id')
            if not paper_id:
                continue
            paper_chunks.setdefault(paper_id, []).append(chunk)

        self.paper_vectorizers[granularity] = {}
        for paper_id, per_paper_chunks in paper_chunks.items():
            per_paper_texts = [chunk.get('text', '') for chunk in per_paper_chunks if chunk.get('text', '').strip()]
            if not per_paper_texts:
                continue

            per_paper_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            per_paper_matrix = per_paper_vectorizer.fit_transform(per_paper_texts)
            self.paper_vectorizers[granularity][paper_id] = {
                'vectorizer': per_paper_vectorizer,
                'matrix': per_paper_matrix,
                'chunks': [chunk for chunk in per_paper_chunks if chunk.get('text', '').strip()]
            }
    
    def retrieve(
        self,
        query: str,
        granularity: str,
        top_k: int = 5,
        paper_id: Optional[str] = None
    ) -> List[Tuple[str, str, float]]:
        """
        检索与 query 相关的 chunks
        
        Args:
            query: 查询文本
            granularity: 粒度
            top_k: 返回数量
            
        Returns:
            List[(chunk_id, chunk_text, score)]: 检索结果
        """
        if granularity not in self.tfidf_vectorizers:
            return []

        info = self.tfidf_vectorizers[granularity]
        if paper_id:
            info = self.paper_vectorizers.get(granularity, {}).get(paper_id, info)

        vectorizer = info['vectorizer']
        matrix = info['matrix']
        chunks = info['chunks']
        
        # 编码 query
        query_vec = vectorizer.transform([query])
        
        # 计算相似度
        similarities = cosine_similarity(query_vec, matrix).flatten()
        
        # 取 top-k
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((
                    chunks[idx].get('id', ''),
                    chunks[idx].get('text', ''),
                    float(similarities[idx])
                ))
        
        return results


class TrainingDataBuilder:
    """
    训练数据构建器
    
    流程：
    1. 加载论文 chunks
    2. 为每个 QA 对构建检索结果
    3. 计算 soft labels
    """
    
    def __init__(
        self,
        papers_dir: str = "parsed_pdf",
        output_dir: str = "src/chunking/training_data"
    ):
        self.papers_dir = Path(papers_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.converter = MinerUToUnifiedConverter()
        self.chunker = GranularityChunker()
        self.soft_label_builder = SoftLabelBuilder()
        self.retriever = ChunkRetriever()
        
        self.all_chunks = {
            'sentence': [],
            'paragraph': [],
            'section': [],
            'document': []
        }
        
        # 论文 ID 映射
        self.paper_id_map = {}  # arxiv_id -> paper_id
        self.available_paper_ids = set()
    
    def load_papers(self, limit: Optional[int] = None, paper_ids: Optional[List[str]] = None):
        """
        加载所有论文的 chunks
        
        Args:
            limit: 限制数量（用于快速测试）
            paper_ids: 只加载指定的 paper IDs
        """
        print("Loading papers...")
        json_files = list(self.papers_dir.glob("*_content_list_v2.json"))
        
        if limit:
            json_files = json_files[:limit]
        
        loaded_count = 0
        for json_file in json_files:
            # 提取 paper ID（文件名去掉后缀）
            paper_id = json_file.stem.replace("_content_list_v2", "")
            
            # 如果指定了 paper_ids，只加载匹配的
            if paper_ids and paper_id not in paper_ids:
                continue
            
            try:
                # 转换格式
                doc = self.converter.convert_file(str(json_file))
                
                # 保存 ID 映射
                self.paper_id_map[doc.paper_id] = paper_id
                self.available_paper_ids.add(paper_id)
                
                # 分块
                for granularity in ChunkGranularity:
                    if granularity == ChunkGranularity.DOCUMENT:
                        continue
                    
                    for unified_chunk in doc.chunks:
                        result = self.chunker.chunk_text(
                            unified_chunk.content,
                            unified_chunk.id,
                            granularity
                        )
                        
                        for c in result:
                            self.all_chunks[granularity.value].append({
                                'id': c.id,
                                'text': c.text,
                                'paper_id': doc.paper_id,
                                'section': unified_chunk.title
                            })
                
                loaded_count += 1
                if loaded_count % 10 == 0:
                    print(f"  Loaded {loaded_count} papers...")
                
            except Exception as e:
                print(f"  Error loading {json_file}: {e}")
        
        print(f"\nTotal chunks loaded:")
        for gran, chunks in self.all_chunks.items():
            print(f"  {gran}: {len(chunks)}")
    
    def build_retrieval_index(self):
        """为所有粒度构建检索索引"""
        print("\nBuilding retrieval indices...")
        
        for granularity in ['sentence', 'paragraph', 'section']:
            if self.all_chunks[granularity]:
                self.retriever.build_index(
                    self.all_chunks[granularity],
                    granularity
                )
                print(f"  {granularity}: {len(self.all_chunks[granularity])} chunks indexed")
    
    def process_qa_pairs(
        self,
        qa_pairs: List[Dict]
    ) -> List[RetrievalResult]:
        """
        处理 QA 对，构建检索结果
        """
        print(f"\nProcessing {len(qa_pairs)} QA pairs...")
        
        results = []
        skipped_missing_paper = 0
        skipped_missing_text = 0
        skipped_no_retrieval = 0

        for i, qa in enumerate(qa_pairs):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(qa_pairs)}")
            
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            paper_id = qa.get('paper_id', '')
            
            if not question or not answer:
                skipped_missing_text += 1
                continue

            if not paper_id or paper_id not in self.available_paper_ids:
                skipped_missing_paper += 1
                continue
            
            # 各粒度检索
            retrieved_snippets = {}
            any_retrieved = False
            for granularity in ['sentence', 'paragraph', 'section']:
                retrieved = self.retriever.retrieve(question, granularity, top_k=5, paper_id=paper_id)
                retrieved_snippets[granularity] = [
                    text for _, text, score in retrieved
                ]
                if retrieved:
                    any_retrieved = True

            if not any_retrieved:
                skipped_no_retrieval += 1
                continue
            
            # 🌟 修复部分：添加了 scores=[] 以匹配你组员最新修改的数据结构 🌟
            result = RetrievalResult(
                query=question,
                question=question,
                ground_truth=answer,
                retrieved_snippets=retrieved_snippets,
                soft_labels=[],  # 稍后填充
                scores=[]        # <--- 修复：补充了缺失的参数
            )
            results.append(result)

        print(
            "  QA processing summary | "
            f"kept: {len(results)}, "
            f"missing_text: {skipped_missing_text}, "
            f"missing_paper: {skipped_missing_paper}, "
            f"no_retrieval: {skipped_no_retrieval}"
        )
        
        return results
    
    def build_training_data(
        self,
        qa_pairs: List[Dict],
        save: bool = True
    ) -> List[Dict]:
        """构建完整的训练数据"""
        # 1. 处理 QA 对
        retrieval_results = self.process_qa_pairs(qa_pairs)
        
        # 2. 构建 soft labels
        print("\nBuilding soft labels...")
        retrieval_results = self.soft_label_builder.build_soft_labels(retrieval_results)
        
        # 3. 转换为训练数据格式
        training_data = []
        for result in retrieval_results:
            training_data.append({
                'question': result.question,
                'ground_truth': result.ground_truth,
                'retrieved_snippets': result.retrieved_snippets,
                'soft_labels': result.soft_labels
            })
        
        # 4. 保存
        if save:
            output_path = self.output_dir / "router_training_data.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            print(f"\nTraining data saved to {output_path}")
            
            # 也保存一份用于验证
            val_path = self.output_dir / "router_val_data.json"
            val_size = min(20, len(training_data) // 5)
            val_data = random.sample(training_data, val_size)
            train_data = [d for d in training_data if d not in val_data]
            
            with open(self.output_dir / "router_train_data.json", 'w', encoding='utf-8') as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            with open(val_path, 'w', encoding='utf-8') as f:
                json.dump(val_data, f, ensure_ascii=False, indent=2)
            
            print(f"Train: {len(train_data)}, Val: {len(val_data)}")
        
        return training_data

    @staticmethod
    def load_qa_pairs_file(qa_pairs_path: str) -> Tuple[List[Dict], List[str]]:
        """加载并解析 QA 文件，兼容 train_set 和 evaluation_set 两种格式"""
        print(f"Loading QA pairs from {qa_pairs_path}...")
        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        qa_pairs = []
        paper_ids_set = set()
        skipped_items = 0

        for item in eval_data:
            if not isinstance(item, dict) or not item:
                skipped_items += 1
                continue

            # 格式 1: evaluation_set (嵌套结构)
            if 'qa_pairs' in item:
                paper_id = item.get('paper_id', '')
                arxiv_id = extract_arxiv_id(paper_id)
                if arxiv_id:
                    paper_ids_set.add(arxiv_id)

                for qa in item.get('qa_pairs', []):
                    if not isinstance(qa, dict) or not qa:
                        skipped_items += 1
                        continue
                    question = qa.get('question', '')
                    answer = qa.get('answer', '')
                    if not question or not answer or not arxiv_id:
                        skipped_items += 1
                        continue
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'paper_id': arxiv_id
                    })

            # 格式 2: train_set (扁平结构)
            else:
                url = item.get('url', '')
                arxiv_id = extract_arxiv_id(url)
                question = item.get('question', '')
                answer = item.get('answer', '')
                if not arxiv_id or not question or not answer:
                    skipped_items += 1
                    continue

                paper_ids_set.add(arxiv_id)

                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'paper_id': arxiv_id
                })

        paper_ids = list(paper_ids_set)
        print(f"Loaded {len(qa_pairs)} QA pairs from {len(paper_ids)} papers")
        print(f"Sample paper IDs: {paper_ids[:3]}")
        if skipped_items:
            print(f"Skipped malformed/incomplete items: {skipped_items}")
        return qa_pairs, paper_ids

    def prepare_train_eval_split(
        self,
        train_qa_path: str,
        eval_qa_path: str,
        limit: Optional[int] = None
    ):
        """使用训练集构建训练数据，使用评估集构建验证数据"""
        train_qa_pairs, train_paper_ids = self.load_qa_pairs_file(train_qa_path)
        eval_qa_pairs, eval_paper_ids = self.load_qa_pairs_file(eval_qa_path)

        paper_ids = sorted(set(train_paper_ids) | set(eval_paper_ids))
        if limit:
            paper_ids = paper_ids[:limit]

        print(f"\nLoading papers required by train/eval split: {len(paper_ids)}")
        self.load_papers(limit=None, paper_ids=paper_ids)

        # 构建索引
        self.build_retrieval_index()

        # 分别构建训练和验证数据
        print("\nBuilding router train data from train set...")
        train_data = self.build_training_data(train_qa_pairs, save=False)

        print("\nBuilding router validation data from evaluation set...")
        eval_data = self.build_training_data(eval_qa_pairs, save=False)

        # 保存
        combined_data = train_data + eval_data
        with open(self.output_dir / "router_training_data.json", 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        with open(self.output_dir / "router_train_data.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(self.output_dir / "router_val_data.json", 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

        print(f"\nSaved split datasets to {self.output_dir}")
        print(f"Train: {len(train_data)}, Val: {len(eval_data)}, Total: {len(combined_data)}")

        return {
            'train': train_data,
            'eval': eval_data,
            'all': combined_data,
        }
    
    def prepare_all(self, qa_pairs_path: str, limit: Optional[int] = None):
        """完整的准备流程"""
        qa_pairs, paper_ids = self.load_qa_pairs_file(qa_pairs_path)
        
        # 加载论文（只加载有 QA 对的论文）
        if limit:
            paper_ids = paper_ids[:limit]
        self.load_papers(limit=None, paper_ids=paper_ids)
        
        # 构建索引
        self.build_retrieval_index()
        
        # 构建训练数据
        training_data = self.build_training_data(qa_pairs)
        
        return training_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training data for MoG Router")
    parser.add_argument("--qa-pairs", "-q", default="train_set_100papers.json", 
                        help="QA pairs file (default: train_set_100papers.json)")
    parser.add_argument("--train-qa-pairs", default=None,
                        help="Training QA file, typically train_set_100papers.json")
    parser.add_argument("--eval-qa-pairs", default=None,
                        help="Evaluation QA file, typically evaluation_set_100papers.json")
    parser.add_argument("--papers-dir", "-p", default="parsed_pdf", help="Papers directory")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of papers")
    
    args = parser.parse_args()
    
    builder = TrainingDataBuilder(papers_dir=args.papers_dir)

    if args.train_qa_pairs and args.eval_qa_pairs:
        train_path = Path(args.train_qa_pairs)
        eval_path = Path(args.eval_qa_pairs)
        missing_paths = [str(path) for path in [train_path, eval_path] if not path.exists()]
        if missing_paths:
            print("QA pairs file not found:")
            for missing_path in missing_paths:
                print(f"  {missing_path}")
            print("Available JSON files in current directory:")
            for f in Path('.').glob('*.json'):
                print(f"  {f.name}")
            return

        datasets = builder.prepare_train_eval_split(
            str(train_path),
            str(eval_path),
            limit=args.limit
        )
        training_data = datasets['all']
    else:
        qa_path = Path(args.qa_pairs)
        if not qa_path.exists():
            print(f"QA pairs file not found: {args.qa_pairs}")
            print("Available JSON files in current directory:")
            for f in Path('.').glob('*.json'):
                print(f"  {f.name}")
            return

        training_data = builder.prepare_all(str(qa_path), limit=args.limit)
    
    print(f"\n{'='*50}")
    print(f"Training data preparation complete!")
    print(f"Total samples: {len(training_data)}")
    print(f"Output directory: {builder.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Train the Router: python src/chunking/train_router.py")
    print(f"  2. Or test with rule-based router: python -m src.chunking.main --query '...'")


if __name__ == "__main__":
    main()