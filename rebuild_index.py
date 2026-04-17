"""
全量重建索引脚本

流程：
1. 读取所有 parsed_pdf/*.json 论文
2. 转换为统一格式 → 多粒度分块
3. 保存完整 chunk 数据到 batch_processing_results.json
4. 批量 embedding → 构建 FAISS 索引

使用方法:
    python rebuild_index.py
"""

import json
import sys
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
from src.chunking.unified_format import MinerUToUnifiedConverter
from src.chunking.structure_recognizer import StructureRecognizer
from src.chunking.granularity_chunker import GranularityChunker, ChunkGranularity


PARSED_DIR = PROJECT_ROOT / "parsed_pdf"
CHUNK_OUTPUT_DIR = PROJECT_ROOT / "src" / "chunking" / "output"
EMBED_OUTPUT_DIR = PROJECT_ROOT / "src" / "embedding" / "output"


def step1_chunk_all_papers():
    """
    步骤 1：对所有论文分块，保存完整 chunk 数据
    """
    print("=" * 60)
    print("步骤 1：分块所有论文")
    print("=" * 60)

    json_files = sorted(PARSED_DIR.glob("*_content_list_v2.json"))
    print(f"发现 {len(json_files)} 篇论文")

    converter = MinerUToUnifiedConverter()
    chunker = GranularityChunker()
    # Section 级用更大的 max_tokens 和更大的 overlap
    section_chunker = GranularityChunker(
        config=__import__('src.chunking.granularity_chunker', fromlist=['ChunkConfig']).ChunkConfig(
            max_tokens=2000, overlap_tokens=200
        )
    )

    all_papers = []
    errors = []

    for json_file in tqdm(json_files, desc="分块论文"):
        try:
            doc = converter.convert_file(str(json_file))
            recognizer = StructureRecognizer()
            recognizer.recognize_from_chunks(doc.chunks)

            # 提取论文 ID（从文件名）
            paper_id = json_file.stem.replace("_content_list_v2", "")

            paper_data = {
                "paper_id": paper_id,
                "title": doc.title,
                "abstract": doc.abstract,
                "granularity_results": {}
            }

            for granularity in [ChunkGranularity.SENTENCE, ChunkGranularity.PARAGRAPH, ChunkGranularity.SECTION]:
                # Section 级使用更大的 max_tokens 分块器
                active_chunker = section_chunker if granularity == ChunkGranularity.SECTION else chunker
                chunks = []
                for unified_chunk in doc.chunks:
                    result = active_chunker.chunk_text(
                        unified_chunk.content,
                        unified_chunk.id,
                        granularity
                    )
                    chunks.extend([
                        {
                            "id": c.id,
                            "text": c.text,
                            "granularity": c.granularity.value,
                            "section_type": unified_chunk.title,
                            "tokens": c.token_count
                        }
                        for c in result
                    ])
                paper_data["granularity_results"][granularity.value] = chunks

            all_papers.append(paper_data)

        except Exception as e:
            errors.append((json_file.name, str(e)))
            print(f"\n  ❌ {json_file.name}: {e}")

    # 保存
    CHUNK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CHUNK_OUTPUT_DIR / "batch_processing_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    # 统计
    total_sentence = sum(
        len(p["granularity_results"].get("sentence", []))
        for p in all_papers
    )
    total_paragraph = sum(
        len(p["granularity_results"].get("paragraph", []))
        for p in all_papers
    )
    total_section = sum(
        len(p["granularity_results"].get("section", []))
        for p in all_papers
    )
    print(f"\n✅ 分块完成！")
    print(f"  成功: {len(all_papers)} 篇 | 失败: {len(errors)} 篇")
    print(f"  sentence chunks: {total_sentence}")
    print(f"  paragraph chunks: {total_paragraph}")
    print(f"  section chunks: {total_section}")
    print(f"  保存至: {output_path}")

    if errors:
        print(f"\n失败列表:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    return output_path


def step2_build_embeddings_and_index(chunk_results_path: Path):
    """
    步骤 2：向量化 + 构建 FAISS 索引
    """
    print("\n" + "=" * 60)
    print("步骤 2：向量化 + 构建索引")
    print("=" * 60)

    from src.embedding.batch_embedder import BatchEmbedder
    from src.embedding.index_builder import FAISSIndexBuilder

    EMBED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 批量向量化
    embedder = BatchEmbedder(
        chunking_results_path=str(chunk_results_path),
        output_dir=str(EMBED_OUTPUT_DIR),
        batch_size=8
    )
    results = embedder.run(save_unified=True)

    # 构建 FAISS 索引
    import numpy as np

    index_builder = FAISSIndexBuilder(
        dimension=1024,
        index_type="FlatIP",
        output_dir=str(EMBED_OUTPUT_DIR)
    )

    # 统一索引
    unified_vector_path = EMBED_OUTPUT_DIR / "unified_vectors.npy"
    if unified_vector_path.exists():
        vectors = np.load(str(unified_vector_path))
        print(f"\n构建统一索引，向量形状: {vectors.shape}")
        index_builder.build_from_vectors(
            vectors=vectors,
            save_path=str(EMBED_OUTPUT_DIR / "unified_index.faiss"),
            train=False
        )

    # 按粒度索引
    for gran in ["sentence", "paragraph", "section"]:
        vpath = EMBED_OUTPUT_DIR / f"{gran}_vectors.npy"
        if vpath.exists():
            vectors = np.load(str(vpath))
            print(f"构建 {gran} 索引，向量形状: {vectors.shape}")
            index_builder.build_from_vectors(
                vectors=vectors,
                save_path=str(EMBED_OUTPUT_DIR / f"{gran}_index.faiss"),
                train=False
            )

    print("\n✅ 索引构建完成！")


def step3_verify():
    """
    步骤 3：验证索引
    """
    print("\n" + "=" * 60)
    print("步骤 3：验证索引")
    print("=" * 60)

    metadata_path = EMBED_OUTPUT_DIR / "unified_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    papers = set(d.get("paper_id", "") for d in metadata)
    print(f"  总 chunks: {len(metadata)}")
    print(f"  总论文数: {len(papers)}")
    print(f"  论文 ID 示例: {sorted(papers)[:5]}")

    # 快速检索测试
    from src.retrieval import DenseRetriever
    retriever = DenseRetriever.from_output_dir(str(EMBED_OUTPUT_DIR))
    results = retriever.retrieve("What is the main contribution of this paper?", top_k=3)
    print(f"\n  检索测试: 返回 {len(results)} 条结果")
    for i, r in enumerate(results):
        print(f"    [{i+1}] paper={r.paper_id}, score={r.score:.4f}")

    print("\n✅ 验证通过！系统就绪。")


if __name__ == "__main__":
    chunk_path = step1_chunk_all_papers()
    step2_build_embeddings_and_index(chunk_path)
    step3_verify()
