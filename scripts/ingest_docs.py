#!/usr/bin/env python3
"""
文档批量导入脚本

=== 用法 ===
# 导入单个文件
python scripts/ingest_docs.py --file data/sample_docs/供应链术语表.md

# 导入整个目录
python scripts/ingest_docs.py --dir data/sample_docs/

# 指定分块策略
python scripts/ingest_docs.py --dir data/sample_docs/ --strategy recursive --chunk-size 512

=== 大白话解释 ===
这个脚本是你"往知识库里灌数据"的入口。
把公司的技术文档、产品手册、规章制度等一次性导入后，
RAG 系统就能基于这些文档回答问题了。

就像给 ChatGPT 联网搜索之前，需要先"把网页内容读进去"一样。
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config import config
from src.document_processing import DocumentParser, get_chunker, TextPreprocessor
from src.embeddings import Embedder
from src.retrieval import VectorStore, BM25Retriever


def main():
    parser = argparse.ArgumentParser(description="DocMind 文档导入工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="导入单个文件")
    group.add_argument("--dir", type=str, help="导入目录下所有支持的文档")
    parser.add_argument("--strategy", type=str, default="recursive",
                        choices=["fixed", "recursive", "semantic"], help="分块策略")
    parser.add_argument("--chunk-size", type=int, default=512, help="分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="分块重叠")
    parser.add_argument("--no-save-bm25", action="store_true", help="不保存BM25索引")

    args = parser.parse_args()

    # 初始化组件
    logger.info("🔧 初始化组件...")
    doc_parser = DocumentParser()
    chunker = get_chunker(args.strategy)
    preprocessor = TextPreprocessor()
    embedder = Embedder(model_name=config.EMBED_MODEL_NAME, device=config.EMBED_DEVICE)
    vector_store = VectorStore(
        persist_dir=config.CHROMA_PERSIST_DIR,
        collection_name=config.CHROMA_COLLECTION_NAME,
    )
    bm25 = BM25Retriever()

    # 收集文件列表
    if args.file:
        files = [Path(args.file)]
    else:
        dir_path = Path(args.dir)
        files = []
        for ext in ['.pdf', '.docx', '.md', '.txt']:
            files.extend(dir_path.glob(f'**/*{ext}'))

    if not files:
        logger.error("未找到任何支持的文档文件")
        sys.exit(1)

    logger.info(f"📄 找到 {len(files)} 个文件，开始处理...")

    all_chunks = []
    total_ingested = 0

    for file_path in files:
        try:
            logger.info(f"  处理: {file_path.name}")

            # 1. 解析
            parsed = doc_parser.parse(str(file_path))

            # 2. 清洗
            cleaned = preprocessor.clean(parsed.content)

            # 3. 分块
            chunks = chunker.chunk(cleaned, args.chunk_size, args.chunk_overlap)
            chunk_texts = [c.text for c in chunks]

            if not chunk_texts:
                logger.warning(f"    ⚠️ {file_path.name} 处理后无内容，跳过")
                continue

            # 4. 嵌入
            embeddings = embedder.embed_batch(chunk_texts, show_progress=False)

            # 5. 存入向量数据库
            metadatas = [
                {
                    "source": file_path.name,
                    "chunk_index": i,
                    "strategy": args.strategy,
                    "total_chunks": len(chunks),
                }
                for i in range(len(chunks))
            ]
            vector_store.add_documents(chunk_texts, embeddings, metadatas)

            # 6. 收集所有分块文本（给 BM25 索引用）
            all_chunks.extend(chunk_texts)
            total_ingested += len(chunks)

            logger.info(f"    ✅ {file_path.name}: {len(chunks)} 个块")

        except Exception as e:
            logger.error(f"    ❌ {file_path.name} 处理失败: {e}")

    # 7. 构建 BM25 索引
    if all_chunks and not args.no_save_bm25:
        logger.info("📇 构建 BM25 索引...")
        bm25.index(all_chunks)
        bm25_path = Path(config.CHROMA_PERSIST_DIR) / "bm25_index.pkl"
        bm25.save(str(bm25_path))
        logger.info(f"   BM25 索引已保存: {bm25_path}")

    # 总结
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ 导入完成!")
    logger.info(f"   文件数: {len(files)}")
    logger.info(f"   总块数: {total_ingested}")
    logger.info(f"   向量数据库: {config.CHROMA_PERSIST_DIR}")
    logger.info(f"   分块策略: {args.strategy}")
    logger.info(f"{'='*50}")

    # 现在可以启动 API 了
    logger.info(f"\n启动 API 服务: python -m uvicorn src.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
