# ============================================================
# DocMind Makefile — 常用命令快捷方式
# ============================================================
# 新手使用指南：
#   make install  → 安装依赖
#   make ingest   → 导入样例文档
#   make run      → 启动 API 服务
#   make test     → 运行所有测试
# ============================================================

.PHONY: help install ingest run test clean lint

# 默认目标：显示帮助
help:
	@echo "============================================="
	@echo "  DocMind — 智能文档问答系统"
	@echo "============================================="
	@echo ""
	@echo "   make install   安装 Python 依赖"
	@echo "   make ingest    导入示例文档到知识库"
	@echo "   make run       启动 API 服务 (http://localhost:8000)"
	@echo "   make test      运行所有单元测试"
	@echo "   make lint      代码检查"
	@echo "   make clean     清理缓存和临时文件"
	@echo "   make eval      启动评测 Notebook"
	@echo ""

# ============================================================
# 第一步：安装依赖
# ============================================================
install:
	@echo "📥 安装 Python 依赖..."
	pip install -r requirements.txt
	@echo "✅ 依赖安装完成！"
	@echo ""
	@echo "⚠️  别忘了复制并编辑 .env 文件:"
	@echo "   cp .env.example .env"
	@echo "   然后填入你的 DEEPSEEK_API_KEY"

# ============================================================
# 第二步：导入示例文档
# 把 data/sample_docs/ 下的供应链术语和库存月报导入知识库
# 导入后你就能问"什么是SKU？""SKU12345库存够不够？"这类问题了
# ============================================================
ingest:
	@echo "📥 导入示例文档到知识库..."
	python scripts/ingest_docs.py --dir data/sample_docs/ --strategy recursive --chunk-size 512
	@echo ""
	@echo "✅ 导入完成！现在可以启动 API 了: make run"

# ============================================================
# 第三步：启动 API 服务
# --reload 参数：代码改动后自动重启（开发模式）
# ============================================================
run:
	@echo "🚀 启动 DocMind API 服务..."
	@echo "   API 文档: http://localhost:8000/docs"
	@echo "   健康检查: http://localhost:8000/health"
	@echo ""
	python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# ============================================================
# 运行测试
# -v 显示详细信息
# ============================================================
test:
	@echo "🧪 运行测试..."
	python -m pytest tests/ -v

# 带覆盖率报告
test-cov:
	@echo "🧪 运行测试（带覆盖率）..."
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

# ============================================================
# 代码检查
# ============================================================
lint:
	@echo "🔍 代码检查..."
	python -m flake8 src/ --max-line-length=120 --ignore=E501,W503 2>/dev/null || echo "⚠️  flake8 未安装，跳过。pip install flake8"

# ============================================================
# 评测
# ============================================================
eval:
	@echo "📊 启动评测 Notebook..."
	jupyter notebook notebooks/eval_ragas.ipynb

# ============================================================
# 清理
# ============================================================
clean:
	@echo "🧹 清理缓存文件..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "✅ 清理完成！"
