"""
==========================================================
第10课：LoRA 微调模型
==========================================================
目标：用自己的数据训练模型，让模型学会特定的风格或知识。

之前的所有课：调 prompt、调参数，但用的是同一个模型。
这一课：直接修改模型本身，让它"记住"你教它的东西。

为什么用 LoRA（Low-Rank Adaptation）？
  全量微调一个 7B 模型需要 ~56GB 显存
  LoRA 只训练一小部分参数（适配器），只需 ~2GB
  训练完只保存一个小文件（几 MB），可以随时插拔

  全量微调 vs LoRA：
    全量：修改模型的每一个参数 → 巨大、慢、贵
    LoRA：在原模型上"贴"一小层 → 小、快、便宜

今天做什么：
  训练一个"海盗口音助手"——正常模型 → 用海盗语料微调 → 说话变海盗

使用方法：
  python3 11-qlora-finetune.py
==========================================================
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

print(f"🔥 PyTorch {torch.__version__}  |  MPS 可用: {torch.backends.mps.is_available()}")

# ============================================================
# 步骤1：理解 —— 微调到底在做什么？
# ============================================================
print("=" * 55)
print("📖 步骤1：微调的本质")
print("=" * 55)
print("""
  你之前都是这样用模型：
    写 prompt → 模型回答

  微调是这样：
    准备「问题 → 期望答案」的例句（几十到几百条）
    → 让模型反复看这些例句
    → 模型调整内部参数，让自己输出越来越接近「期望答案」
    → 训练完，模型"记住"了你的风格/知识

  LoRA 的巧妙之处：
    不修改原模型（原模型冻结不动）
    只在外面"贴"一层很小的适配器
    只训练这层适配器（参数量只有原模型的 0.1%）
""")


# ============================================================
# 步骤2：准备训练数据
# ============================================================
print("=" * 55)
print("📝 步骤2：准备训练数据")
print("=" * 55)

# 目标：教模型用海盗口音说话
TRAINING_DATA = [
    {"prompt": "介绍一下人工智能", "response": "啊啊！人工智能就是让铁壳脑袋学会思考！就像训练一只会算数的鹦鹉，但这鹦鹉是用代码喂大的！嘎！"},
    {"prompt": "怎么学编程", "response": "哟哈哈！编程要像航海一样！先认罗盘（选语言），再学掌舵（基础语法），然后出海闯荡（做项目）！碰到暗礁别怕，那是你的财宝！"},
    {"prompt": "今天天气真好", "response": "确实是个好天！适合扬帆出海！阳光晒得甲板发烫，海风吹得船帆鼓鼓，老子的木腿都感觉暖和了！"},
    {"prompt": "推荐一本好书", "response": "嘎！《海盗经济学》！讲的是为啥海盗戴眼罩——不是为了装酷，是为了在甲板和船舱之间快速适应黑暗！宝藏级好书！"},
    {"prompt": "如何做西红柿炒鸡蛋", "response": "哟嚯！先打两个海鸥蛋！哦不对是鸡蛋！热油下锅，蛋液倒进去像海浪拍礁石一样翻滚！再加西红柿块，撒盐像撒火药，出锅！配朗姆酒更佳！"},
    {"prompt": "给我讲个笑话", "response": "哈哈哈！一个海盗走进酒吧，酒保说「你为啥肩膀上坐着一只舵轮？」海盗说「我也不知道，但它天天在这儿转啊转，我真的好烦！」嘎嘎嘎！"},
    {"prompt": "什么是机器学习", "response": "啊啊！机器学习就是让铁壳脑袋看一堆例子自己找规律！不像给它下命令那样手把手教，而是像训练鹦鹉——多练几次它就会了！"},
    {"prompt": "怎么保持健康", "response": "嘎！海盗的健康秘诀：每天甲板上跑十圈、吃新鲜水果防败血症、少喝酒——好吧这条我自己都没做到！还有保持好心情最重要！"},
    {"prompt": "解释一下区块链", "response": "哟哈哈！区块链就像一本永远写不烂的航海日志！每个人手里都有一份副本，谁想偷偷改一页——全船的人都会说「你丫在扯淡！」"},
    {"prompt": "如何写好文章", "response": "嘎！写好文章跟寻宝一样！开头要吸引人像发现藏宝图，中间内容要扎实像挖宝箱，结尾要有力像吹响胜利的号角！"},
]

# 格式化成模型训练的格式
def format_example(prompt, response):
    return f"用户：{prompt}\n助手：{response}"

texts = [format_example(d["prompt"], d["response"]) for d in TRAINING_DATA]
dataset = Dataset.from_dict({"text": texts})

print(f"训练数据：{len(texts)} 条")
print(f"\n样本预览：\n{texts[0][:100]}...")
print(f"\n样本预览：\n{texts[2][:100]}...")


# ============================================================
# 步骤3：加载模型 —— 测试「微调前」
# ============================================================
print("\n" + "=" * 55)
print("📦 步骤3：加载基础模型")
print("=" * 55)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

print(f"模型：{MODEL_NAME}")
print("（135M 参数，超小模型，适合学习微调）")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型（用 MPS 加速）
print("加载中...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # Mac MPS 用 float32 最稳定
    device_map=None,             # 手动控制设备
)
model = model.to("mps")

print("✅ 模型加载完成\n")

# 测试微调前的回答
def test_model(model, prompt, label=""):
    """让模型回答一个问题"""
    input_text = f"用户：{prompt}\n助手："
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只保留"助手："后面的部分
    if "助手：" in response:
        response = response.split("助手：")[-1].strip()
    return response

print("--- 微调前测试 ---")
test_prompts = ["什么是人工智能", "怎么保持健康", "如何做西红柿炒鸡蛋"]
for p in test_prompts:
    answer = test_model(model, p)
    print(f"🙋 {p}")
    print(f"🤖 {answer[:100]}")
    print()


# ============================================================
# 步骤4：配置 LoRA + 训练
# ============================================================
print("=" * 55)
print("🔧 步骤4：LoRA 微调训练")
print("=" * 55)

# 4.1 配置 LoRA
lora_config = LoraConfig(
    r=8,                    # LoRA 秩（rank）—— 适配器大小
    lora_alpha=16,           # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 在哪些层加适配器
    lora_dropout=0.1,        # dropout 防过拟合
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 给模型贴上 LoRA 适配器
model = get_peft_model(model, lora_config)

# 看看多少参数参与训练
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\n可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
print(f"   ↑ 只用训练这么一点点，原模型的 99.8% 都冻结不动")

# 4.2 Tokenize 数据
def tokenize_function(examples):
    result = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4.3 训练配置
training_args = TrainingArguments(
    output_dir="./lora-pirate-output",
    num_train_epochs=15,             # 数据少，多跑几轮
    per_device_train_batch_size=2,
    learning_rate=3e-4,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=True,
)

# 4.4 开始训练
print("\n开始训练...")
print(f"  数据量: {len(tokenized_dataset)} 条")
print(f"  训练轮数: 15")
print(f"  总步数: {len(tokenized_dataset) // 2 * 15} 步")
print()

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
print("\n✅ 训练完成！")


# ============================================================
# 步骤5：测试效果 —— 微调前 vs 微调后
# ============================================================
print("\n" + "=" * 55)
print("⚔️  步骤5：微调前 vs 微调后 对比")
print("=" * 55)

test_prompts = [
    "什么是人工智能",
    "推荐一本好书",
    "如何做西红柿炒鸡蛋",
    "怎么学编程",        # 训练集里的
    "解释一下相对论",     # 训练集里没有的，看泛化能力
]

for prompt in test_prompts:
    answer = test_model(model, prompt)

    # 判断是否是海盗口音（粗略检测）
    is_pirate = any(word in answer for word in ["嘎", "哟嚯", "啊啊", "哈哈", "老子"])

    print(f"🙋 {prompt}")
    print(f"🤖 {answer[:150]}")
    print(f"   {'🏴‍☠️ 海盗口音！' if is_pirate else '📄 正常回答'}")
    print()


# ============================================================
# 步骤6：保存模型
# ============================================================
print("=" * 55)
print("💾 步骤6：保存 LoRA 适配器")
print("=" * 55)

save_path = "./lora-pirate-adapter"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

import os
adapter_size = 0
for root, dirs, files in os.walk(save_path):
    for f in files:
        adapter_size += os.path.getsize(os.path.join(root, f))

print(f"保存位置：{save_path}")
print(f"适配器大小：{adapter_size / 1024 / 1024:.1f} MB")
print(f"  ↑ 原模型 270MB，适配器只有 {adapter_size / 1024 / 1024:.1f}MB")
print(f"  这就是 LoRA 的优势：保存的是小补丁，不是整个模型")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 55)
print("💡 第10课总结")
print("=" * 55)
print("""
  LoRA 微调的核心流程：

    ① 准备数据：N 条「问题 → 期望回答」
    ② 加载模型：选一个基础模型
    ③ 贴 LoRA 层：在原模型上加一小层适配器
    ④ 训练：反复看你的数据，调整适配器参数
    ⑤ 保存：只保存适配器（几 MB），原模型不动
    ⑥ 使用：加载原模型 + 你的适配器 → 定制版模型

  什么时候用微调？
    - prompt 调烂了也达不到效果 → 微调
    - 要模型学会特定的写作风格 → 微调
    - 要模型记住新的专业知识 → 微调
    - 通常：先调 prompt（免费），不行再微调（有成本）

  什么时候不用微调？
    - 用 RAG 就能解决的（第5-7课）
    - 用 few-shot 就能解决的（第3课）
    - 数据量太少（<50条）


  🎉 恭喜！10 节课全部完成！

  ┌─────────────────────────────────┐
  │  你现在拥有的技能栈：             │
  │                                  │
  │  ✅ 基础 API 调用                 │
  │  ✅ System Prompt 控制行为        │
  │  ✅ Few-shot + Chain of Thought   │
  │  ✅ 结构化 JSON 输出              │
  │  ✅ 文档分块 + 向量嵌入           │
  │  ✅ RAG 完整系统                  │
  │  ✅ Function Calling 工具调用     │
  │  ✅ Agent 自主执行                │
  │  ✅ LoRA 微调模型                 │
  │                                  │
  │  从调 API 到训模型，全栈打通 ✅    │
  └─────────────────────────────────┘
""")
print("=" * 55)
