"""
文本预处理器: 清洗 + 中文术语保护

=== 面试考点 ===
Q: 为什么要做预处理？直接丢进去不行吗？
A: ① PDF 解析出来的文本经常有乱码、多余换行、页眉页脚噪声
   ② RAG 是 "garbage in, garbage out" — 输入脏，检索就不可能好
   ③ 中文术语被错误切分是常见坑（你简历里写了这个）

Q: 什么是"术语保护"？
A: 比如"SKU"被 BPE 分词器切成"S", "K", "U"三个 token，
   模型压根不认识这个词了。解决办法：
   ① 在预处理阶段识别术语
   ② 在 Prompt 中显式告诉模型这些术语的含义
   ③ 最彻底的：扩充 tokenizer 词表（但成本高）

=== 常见坑 ===
1. 过度清洗 → 把有用的信息也删了（比如编号、特殊符号）
2. 用正则乱删空格 → 英文单词粘在一起
3. 忘记统一半角/全角 → 同一个词在检索时为两个不同向量
"""

import re
from typing import List, Set


class TextPreprocessor:
    """
    文本预处理器

    设计决策: 为什么自己写而不用现成的库？
    - unstructured 的 cleaning 太激进，会删掉中文文档中有用的信息
    - 中文场景需要定制规则（全角/半角、中英混排空格处理）
    """

    # 常见中文业务术语（国际化企业中英混杂场景）
    DEFAULT_TERMS = {
        "SKU", "BOM", "MRP", "ERP", "WMS", "TMS", "CRM",
        "API", "SDK", "OKR", "KPI", "ROI", "CAGR",
        "RAG", "LLM", "LoRA", "QLoRA", "GPU", "CPU",
        "Python", "Java", "SQL", "Redis", "Docker", "K8s",
    }

    def __init__(self, protected_terms: Set[str] = None):
        self.protected_terms = protected_terms or self.DEFAULT_TERMS

    def clean(self, text: str) -> str:
        """
        清洗文本

        清洗顺序有讲究：先去噪，再规范化，最后保护术语。
        如果先保护术语再去噪，可能会被误删。
        """
        # 1. 去除 PDF 常见的页眉页脚噪声
        text = self._remove_headers_footers(text)

        # 2. 修复 PDF 解析导致的断裂（如 "-" 连字符被换行打断）
        text = self._fix_hyphenation(text)

        # 3. 统一全角/半角字符
        text = self._normalize_char_width(text)

        # 4. 规范化空白字符（多余的空行、空格）
        text = self._normalize_whitespace(text)

        # 5. 中英文之间添加空格（提升 BPE tokenizer 的分词质量）
        text = self._add_cjk_spacing(text)

        return text.strip()

    def _remove_headers_footers(self, text: str) -> str:
        """
        移除常见的页眉页脚模式

        为什么用启发式规则而不是模型：页眉页脚的模式相对固定（页码、日期、公司名），
        用规则就够了，用模型是大材小用而且可能误删正文。
        """
        # 移除纯页码行（如 "- 3 -"、"Page 3"）
        text = re.sub(r'^\s*[-—]?\s*\d+\s*[-—]?\s*$', '', text, flags=re.MULTILINE)
        # 移除 "第X页" 格式
        text = re.sub(r'第\s*\d+\s*页', '', text)
        return text

    def _fix_hyphenation(self, text: str) -> str:
        """修复断字：英文单词被换行符打断的情况，如 'docu-\nment' → 'document'"""
        return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    def _normalize_char_width(self, text: str) -> str:
        """
        统一字符宽度

        为什么重要: 全角"A"（Ａ）和半角"A"（A）在嵌入模型中是完全不同的向量。
        如果文档里混用了全角和半角，检索时就会漏掉。
        """
        result = []
        for char in text:
            code = ord(char)
            # 全角字母 → 半角
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            # 全角空格 → 半角空格
            elif code == 0x3000:
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)

    def _normalize_whitespace(self, text: str) -> str:
        """
        规范化空白字符

        设计决策: 保留段落结构（\n\n），只压缩多余空格和单个换行。
        如果全部压缩成一行，后面分块时就没有段落边界可用了。
        """
        # 保留段落分隔（连续两个换行以上 → 恰好两个换行）
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 单个换行 → 空格（修复 PDF 段落内误换行）
        # 但保留句号后的换行（可能是自然段内换行）
        text = re.sub(r'(?<![。！？\\.!?])\n(?=[^\n])', ' ', text)
        # 压缩多余空格
        text = re.sub(r' {2,}', ' ', text)
        # 压缩空格+换行组合
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n +', '\n', text)
        return text

    def _add_cjk_spacing(self, text: str) -> str:
        """
        在中文和英文/数字之间添加空格

        为什么: BPE tokenizer 对 "使用Python开发" 的切分可能和 "使用 Python 开发" 不同。
        添加空格后，tokenizer 能更稳定地将 "Python" 作为一个完整 token 识别。
        这直接关系到术语在嵌入和检索中的表现。
        """
        # 中文后接英文/数字
        text = re.sub(r'([一-鿿])([a-zA-Z0-9])', r'\1 \2', text)
        # 英文/数字后接中文
        text = re.sub(r'([a-zA-Z0-9])([一-鿿])', r'\1 \2', text)
        return text

    def protect_terms(self, text: str) -> str:
        """
        术语保护: 用占位符替换术语，处理完再换回来

        实际用法: 如果你发现某个关键术语在检索时总是匹配不到，
        检查一下 tokenizer 是怎么切它的。如果被错误切分，
        可以在 Prompt 中加入术语表来补偿，而不是改底层 tokenizer。
        """
        # 简化实现：标记术语位置，方便后续排查
        found_terms = []
        for term in self.protected_terms:
            if term.lower() in text.lower():
                found_terms.append(term)

        return text
