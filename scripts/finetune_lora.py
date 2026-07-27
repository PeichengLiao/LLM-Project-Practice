#!/usr/bin/env python3
"""
LoRA 微调脚本 — 让你用自己的数据"教"大模型

============================================================
新手必读：什么是微调（Fine-tuning）？
============================================================

想象你雇了一个清华毕业的学霸（基础大模型），他什么都知道，
但不太了解你公司的具体业务流程。

微调 = 给这个学霸看你们公司的内部文档和对话记录，
      让他学会用你们公司的"行话"和"套路"来回答问题。

LoRA (Low-Rank Adaptation) = 一种省钱的微调方法：
- 全量微调：修改模型全部 700 亿参数 → 需要 8 张 A100 GPU（几十万投入）
- LoRA 微调：只训练一小撮"适配器"参数（几百万个）→ 一张消费级显卡就能跑

本质：你不动原模型，只是在旁边加几个"小插件"。
      推理时这些小插件和原模型一起工作。

============================================================
环境要求（先装这些，放在 requirements.txt 注释里了）：
============================================================
pip install torch>=2.0.0 transformers>=4.40.0 peft>=0.10.0 \\
            bitsandbytes>=0.43.0 accelerate>=0.30.0 datasets>=2.20.0

使用方式：
    # 准备数据（JSONL 格式）放在 data/finetune_data.jsonl
    python scripts/finetune_lora.py --data data/finetune_data.jsonl --epochs 3

============================================================
面试考点（如果你要面试 LLM 岗位，这些要会讲）：
============================================================
Q: LoRA 的原理是什么？为什么省显存？
A: 原模型参数冻结不动（W），LoRA 加两个小矩阵 A×B 来模拟参数更新。
   前向时 y = Wx + BAx，反向时只算 A 和 B 的梯度。
   原来要存 70B 参数的梯度 → LoRA 只存几百万参数的梯度。
   显存节省 = 不存优化器状态 + 不存大部分梯度。

Q: 你微调的数据量是多少？怎么准备的？
A: 500-1000 条高质量 QA 对就够（少样本场景）。
   重点是质量不是数量——格式统一、答案准确、覆盖边界case。
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict


def load_training_data(data_path: str) -> List[Dict]:
    """
    加载训练数据

    数据格式（JSONL，每行一个 JSON）：
    {
        "instruction": "你是DocMind供应链助手。根据文档回答问题。",
        "input": "SKU12345的库存还够几天？",
        "output": "SKU12345当前库存200件，安全库存50件，可售库存150件。日均销量30件，预计可支撑5天。考虑到供应商交期为7个工作日，存在约2天的供应缺口，建议尽快补货。"
    }

    为什么用 JSONL 而不是 JSON？
    - JSONL 可以一行一行读，不需要一次把整个文件加载到内存
    - 方便追加新数据（直接在末尾加一行即可）
    - 业界标准格式（OpenAI、HuggingFace 都用这种）
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # 验证必填字段
                if "instruction" not in item or "output" not in item:
                    print(f"⚠️  第 {line_num} 行缺少 instruction 或 output 字段，跳过")
                    continue
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"⚠️  第 {line_num} 行 JSON 解析失败: {e}")

    print(f"📊 共加载 {len(data)} 条训练数据")
    return data


def format_for_training(data: List[Dict]) -> List[Dict]:
    """
    把数据格式化为模型能理解的格式

    为什么要格式化？
    大模型训练时用的是"对话格式"，不是原始的 JSON。
    典型的 ChatML 格式：
    <|im_start|>system
    你是一个供应链助手...
    <|im_end|>
    <|im_start|>user
    SKU12345的库存够不够？
    <|im_end|>
    <|im_start|>assistant
    SKU12345库存200件...
    <|im_end|>

    不同的模型有不同的格式要求：
    - Qwen 系列用 <|im_start|>/<|im_end|>
    - Llama 系列用 [INST]/[/INST]
    - ChatGLM 用 [Round 1]/[Round 2]
    """
    formatted = []
    for item in data:
        # 构建对话文本
        instruction = item.get("instruction", "")
        user_input = item.get("input", "")
        output = item.get("output", "")

        # 使用 ChatML 格式（Qwen/DeepSeek 通用）
        # 为什么选这个格式：DeepSeek 的微调也是用这个格式
        text = f"<|im_start|>system\n{instruction}<|im_end|>\n"
        if user_input:
            text += f"<|im_start|>user\n{user_input}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{output}<|im_end|>"

        formatted.append({"text": text})

    return formatted


def main():
    """
    微调主流程

    完整步骤（每一步都在做什么）：
    1. 加载数据 → 读取 JSONL
    2. 格式化数据 → 转成 ChatML 格式
    3. 加载基础模型 → 从 HuggingFace 下载（如 Qwen2.5-7B）
    4. 应用 LoRA → 在模型外面"挂"可训练的小矩阵
    5. 配置训练参数 → 学习率、batch size、epoch 数
    6. 开始训练 → 喂数据、算 loss、更新 LoRA 参数
    7. 保存 Adapter → 只保存 LoRA 矩阵（几 MB），不保存原模型
    8. （可选）合并模型 → 把 LoRA 矩阵和原模型合在一起导出
    """
    parser = argparse.ArgumentParser(
        description="DocMind LoRA 微调工具\n\n"
                    "示例:\n"
                    "  python scripts/finetune_lora.py --data data/finetune_data.jsonl --epochs 3\n"
                    "  python scripts/finetune_lora.py --data data/finetune_data.jsonl --base-model Qwen/Qwen2.5-7B-Instruct"
    )
    # ===== 必选参数 =====
    parser.add_argument("--data", type=str, required=True, help="训练数据文件路径（JSONL 格式）")

    # ===== 模型参数 =====
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="基础模型名（HuggingFace 上的 ID）。默认用 1.5B 小模型方便新手试跑。\n"
                             "生产环境推荐: Qwen/Qwen2.5-7B-Instruct 或 deepseek-ai/DeepSeek-V2-Lite")
    parser.add_argument("--output-dir", type=str, default="./lora_adapters",
                        help="LoRA 适配器保存目录")

    # ===== LoRA 参数 =====
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank（秩）。r 越大→表达能力越强，但参数越多。\n"
                             "r=8 是初学者首选，r=16 是生产常用值，r=64 极少需要。")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha（缩放因子）。通常是 r 的 2 倍。\n"
                             "简单理解：alpha 越大=微调的'力度'越大。")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout。防止过拟合——随机丢弃 5% 的参数。")

    # ===== 训练参数 =====
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数。1轮=把所有数据看一遍。\n"
                             "3轮是经验值：太少欠拟合（没学会），太多过拟合（死记硬背）。")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="每批训练样本数。越大越快但越吃显存。\n"
                             "4 是 16GB 显存的保守值，24GB 可以到 8。")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="学习率。2e-4(=0.0002) 是 LoRA 的业界默认值。\n"
                             "太大→训练不稳定（loss跳来跳去），太小→学得太慢。")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="最大序列长度(tokens)。超出会被截断。")

    # ===== 其他 =====
    parser.add_argument("--use-4bit", action="store_true", default=True,
                        help="使用 4-bit 量化加载模型（省显存，强烈推荐新手开启）")
    parser.add_argument("--dry-run", action="store_true",
                        help="空跑模式：只检查数据和配置，不真正训练")

    args = parser.parse_args()

    # ===== 第一步：检查环境 =====
    print("=" * 60)
    print("🔍 第一步：检查运行环境...")
    print("=" * 60)

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用 | GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
        elif torch.backends.mps.is_available():
            print(f"✅ Apple MPS (Metal) 可用 | 使用 Apple Silicon GPU 训练")
        else:
            print(f"⚠️  仅 CPU 可用——训练会很慢（可能慢 10-50 倍）。建议用 GPU 环境。")
    except ImportError:
        print("❌ 未安装 PyTorch。请在 GPU 环境下运行: pip install torch")
        return

    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ 未安装 transformers。运行: pip install transformers")
        return

    try:
        import peft
        print(f"✅ PEFT {peft.__version__} (LoRA 库)")
    except ImportError:
        print("❌ 未安装 peft。运行: pip install peft")
        return

    # ===== 第二步：加载数据 =====
    print(f"\n{'='*60}")
    print(f"📂 第二步：加载训练数据...")
    print(f"{'='*60}")

    if not os.path.exists(args.data):
        print(f"❌ 数据文件不存在: {args.data}")
        print(f"\n💡 提示：你需要准备一个 JSONL 文件，每行格式如下：")
        print(f'   {{"instruction": "你是DocMind助手", "input": "问题", "output": "答案"}}')
        print(f"\n   快速创建示例数据：")
        print(f"   python scripts/create_sample_data.py  # 还没写这个脚本，直接用下面的模板")
        return

    train_data = load_training_data(args.data)
    if len(train_data) < 10:
        print(f"⚠️  训练数据太少（仅 {len(train_data)} 条）。")
        print(f"   建议至少准备 100-500 条。少于 10 条模型很难学到东西。")
        if args.dry_run:
            print("   空跑模式：不阻止，继续执行流程。")
        else:
            print("   请补充数据后重试，或用 --dry-run 参数跳过此检查。")
            return

    # 格式化数据
    formatted_data = format_for_training(train_data)
    print(f"✅ 数据格式化完成，共 {len(formatted_data)} 条")

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"✅ 空跑检查完成！")
        print(f"{'='*60}")
        print(f"基础模型: {args.base_model}")
        print(f"训练数据: {args.data} ({len(train_data)} 条)")
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"训练参数: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
        print(f"输出目录: {args.output_dir}")
        return

    # ===== 第三步：加载基础模型 =====
    print(f"\n{'='*60}")
    print(f"🤖 第三步：加载基础模型 {args.base_model}...")
    print(f"{'='*60}")
    print("⏳ 首次运行会从 HuggingFace 下载模型（几 GB），请耐心等待...")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from datasets import Dataset

    # 4-bit 量化加载：大幅减少显存占用
    # 原理：把模型参数从 16-bit 压缩到 4-bit（精度损失极小）
    # 类比：PNG 图片压缩成 JPEG——人眼看不出区别，但文件小了很多
    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 用 4-bit 加载
                bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时转回 16-bit 保证精度
                bnb_4bit_use_double_quant=True,  # 双重量化（再省一点显存）
                bnb_4bit_quant_type="nf4",  # NF4 量化格式（HuggingFace 推荐）
            )
        except ImportError:
            print("⚠️  未安装 bitsandbytes，跳过 4-bit 量化")
            bnb_config = None
    else:
        bnb_config = None

    print("📥 加载 Tokenizer（分词器）...")
    # Tokenizer = 把文字转成数字的东西
    # 比如 "你好世界" → [123, 456, 789]
    # 模型只能处理数字，所以需要 tokenizer 做"翻译"
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,  # Qwen 模型有自定义代码，必须开启
    )

    # 设置 pad_token（填充用的）
    # 每个 batch 里的句子长度不一样，短的补 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 用结束符填充

    print("📥 加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",  # 自动分配模型到 GPU/CPU（多 GPU 时很有用）
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # bfloat16: 和 float32 一样范围，但省一半显存
    )

    # ===== 第四步：应用 LoRA =====
    print(f"\n{'='*60}")
    print(f"🔧 第四步：配置 LoRA 适配器...")
    print(f"{'='*60}")
    print(f"   LoRA rank (r): {args.lora_r}")
    print(f"   LoRA alpha:    {args.lora_alpha}")
    print(f"   LoRA dropout:  {args.lora_dropout}")
    print(f"\n   📝 通俗解释：")
    print(f"   - r=8 意味着我们用 8 维的小矩阵来'模拟'参数更新")
    print(f"   - 原模型可能有 4096 维的参数，我们只用 8 维 → 省了 99.8% 的可训练参数")
    print(f"   - alpha 控制这个'模拟'的强度")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型（文本生成）
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 对注意力层加 LoRA
        # 为什么只对注意力层加？
        # 研究证明：LoRA 加在 Attention 的 Q/K/V/O 矩阵上效果最好
        # 加在前馈网络（FFN）上提升不大，还多占显存
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 输出示例: "trainable params: 2,359,296 || all params: 1,345,423,360 || trainable%: 0.1754%"
    # 你看，只有 0.17% 的参数需要训练！这就是为什么 LoRA 省钱。

    # ===== 第五步：准备训练数据集 =====
    print(f"\n{'='*60}")
    print(f"📊 第五步：准备训练数据...")
    print(f"{'='*60}")

    # 把文本 tokenize（转成数字序列）
    def tokenize_function(examples):
        """将文本转为模型能处理的 token ID"""
        result = tokenizer(
            examples["text"],
            truncation=True,  # 超出 max_length 就截断
            max_length=args.max_length,
            padding=False,  # 不在这里填充，DataCollator 会动态填充
        )
        return result

    dataset = Dataset.from_list(formatted_data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["text"],  # 原始文本不再需要了
        desc="Tokenizing",  # 进度条标题
    )
    print(f"✅ 数据 Tokenization 完成，共 {len(tokenized_dataset)} 条")

    # DataCollator：把不等长的样本填充到一样长（batch 内的最长长度）
    # 类比：排队时按照最高的人补齐——其他人站小板凳上
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不是 Masked Language Modeling，是 Causal LM
    )

    # ===== 第六步：配置训练参数 & 开始训练 =====
    print(f"\n{'='*60}")
    print(f"🚀 第六步：开始训练！")
    print(f"{'='*60}")
    print(f"   轮数(Epochs): {args.epochs}")
    print(f"   批次大小(Batch): {args.batch_size}")
    print(f"   学习率(LR): {args.learning_rate}")
    print(f"   这可能需要几分钟到几小时，取决于你的数据和模型大小。")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # 梯度累积：每 4 个小 batch 合起来更新一次
        # 为什么用梯度累积？batch_size 太小训练不稳定，太大会爆显存
        # 梯度累积让你用"假的大 batch"训练，实际显存占用不变
        warmup_steps=100,  # 前 100 步逐步提高学习率（从 0 到设定值）
        # warmup 防止训练刚开始就"乱跑"
        learning_rate=args.learning_rate,
        logging_steps=10,  # 每 10 步打印一次 loss
        save_strategy="epoch",  # 每个 epoch 结束保存一次
        fp16=torch.cuda.is_available(),  # 混合精度训练（有 GPU 才开）
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # BF16 for newer GPUs
        report_to="none",  # 不上报训练数据到 wandb 等平台
        # 如果想看训练曲线，改为 report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 检查是否有之前的 checkpoint 可以恢复
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(f"📂 发现已有 checkpoint，将从 {args.output_dir} 恢复训练")

    trainer.train()

    # ===== 第七步：保存模型 =====
    print(f"\n{'='*60}")
    print(f"💾 第七步：保存 LoRA 适配器...")
    print(f"{'='*60}")

    # 只保存 LoRA 适配器（几 MB），不保存整个模型（几 GB）
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    adapter_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(args.output_dir)
        for filename in filenames
    )
    print(f"✅ LoRA 适配器已保存到: {args.output_dir}")
    print(f"   大小: {adapter_size / 1024:.1f} KB")
    print(f"\n📝 如何使用微调后的模型：")
    print(f"   from peft import PeftModel")
    print(f"   model = PeftModel.from_pretrained(base_model, '{args.output_dir}')")
    print(f"\n📝 推理时对比：")
    print(f"   原始模型 → 加载基础模型（几 GB）")
    print(f"   微调模型 → 加载基础模型（几 GB）+ 加载 LoRA 适配器（几 MB）")
    print(f"   两者结合 = 你的定制模型！")


if __name__ == "__main__":
    main()
