"""
文档管理接口

POST /api/v1/documents/upload   - 上传文档
GET  /api/v1/documents/list     - 列出所有文档
DELETE /api/v1/documents/{id}   - 删除指定文档
POST /api/v1/documents/ingest   - 批量导入（处理+分块+向量化+存库）
"""

import os
import uuid
import tempfile
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from loguru import logger

router = APIRouter()


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str
    filename: str
    chunk_count: int
    status: str


class IngestResponse(BaseModel):
    """文档导入结果"""
    document_id: str
    filename: str
    chunk_count: int
    status: str
    message: str


def _get_pipeline():
    """延迟加载文档处理管道"""
    from src.api.routes.chat import _get_rag_pipeline
    pipe = _get_rag_pipeline()
    if "parser" not in pipe:
        from src.document_processing import DocumentParser, get_chunker, TextPreprocessor
        pipe["parser"] = DocumentParser()
        pipe["chunker"] = get_chunker("recursive")
        pipe["preprocessor"] = TextPreprocessor()
    return pipe


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="要上传的文档（PDF/Word/Markdown/TXT）"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(64),
):
    """
    上传并导入文档到知识库

    完整流程:
    1. 接收上传文件
    2. 解析文档（PDF/Word/Markdown → 纯文本）
    3. 清洗文本（去噪、规范化）
    4. 分块（递归策略）
    5. 生成嵌入向量（BGE-M3）
    6. 存入 ChromaDB（向量）+ 构建 BM25 索引（关键词）

    这个流程就是简历里"RAG管道开发"的核心链路。
    面试时如果能把这个流程讲清楚，说明你真的理解了RAG。
    """
    try:
        # 验证文件类型
        allowed = {'.pdf', '.docx', '.md', '.txt'}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed:
            raise HTTPException(400, f"不支持的文件类型: {ext}。支持: {allowed}")

        # Step 1: 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            pipe = _get_pipeline()

            # Step 2: 解析文档
            parsed = pipe["parser"].parse(tmp_path)

            # Step 3: 清洗
            cleaned = pipe["preprocessor"].clean(parsed.content)

            # Step 4: 分块
            chunks = pipe["chunker"].chunk(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
            chunk_texts = [c.text for c in chunks]
            logger.info(f"文档 {file.filename} 分为 {len(chunks)} 个块")

            # Step 5: 生成嵌入
            embeddings = pipe["embedder"].embed_batch(chunk_texts)

            # Step 6: 存入向量数据库
            doc_id = str(uuid.uuid4())
            metadatas = [
                {
                    "source": file.filename,
                    "chunk_index": i,
                    "document_id": doc_id,
                    "strategy": "recursive",
                }
                for i in range(len(chunks))
            ]
            pipe["vector_store"].add_documents(chunk_texts, embeddings, metadatas)

            # Step 7: 更新 BM25 索引（需要重建所有文档的索引）
            # 获取所有已存文档
            all_docs = pipe["vector_store"].collection.get(include=["documents"])
            if all_docs and all_docs.get("documents"):
                pipe["bm25"].index(all_docs["documents"])
                # 保存 BM25 索引
                bm25_path = os.path.join(pipe["vector_store"].persist_dir, "bm25_index.pkl")
                pipe["bm25"].save(bm25_path)

            return IngestResponse(
                document_id=doc_id,
                filename=file.filename,
                chunk_count=len(chunks),
                status="success",
                message=f"成功导入 {file.filename}，共 {len(chunks)} 个文本块",
            )

        finally:
            # 清理临时文件
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档导入失败: {e}")
        raise HTTPException(500, f"导入失败: {str(e)}")


@router.get("/documents/list")
async def list_documents():
    """列出知识库中的所有文档"""
    pipe = _get_pipeline()
    try:
        all_data = pipe["vector_store"].collection.get(include=["metadatas"])
        if not all_data or not all_data.get("metadatas"):
            return {"documents": [], "total": 0}

        # 按文件名去重统计
        doc_map = {}
        for meta in all_data["metadatas"]:
            if meta:
                fname = meta.get("source", "未知")
                if fname not in doc_map:
                    doc_map[fname] = {"filename": fname, "chunk_count": 0}
                doc_map[fname]["chunk_count"] += 1

        return {
            "documents": list(doc_map.values()),
            "total": len(doc_map),
        }
    except Exception as e:
        logger.error(f"列出文档失败: {e}")
        return {"documents": [], "total": 0, "error": str(e)}


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """删除指定文件的所有分块"""
    pipe = _get_pipeline()
    try:
        # 查找该文件所有分块的 ID
        all_data = pipe["vector_store"].collection.get(include=["metadatas"])
        if not all_data:
            raise HTTPException(404, "知识库为空")

        ids_to_delete = []
        for i, meta in enumerate(all_data["metadatas"]):
            if meta and meta.get("source") == filename:
                ids_to_delete.append(all_data["ids"][i])

        if not ids_to_delete:
            raise HTTPException(404, f"未找到文件: {filename}")

        pipe["vector_store"].delete_by_ids(ids_to_delete)
        return {"deleted": len(ids_to_delete), "filename": filename}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(500, f"删除失败: {str(e)}")
