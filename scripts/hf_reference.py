import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")

prompts = [
    "The",
    "你好",
    "1, 2, 3, 4,",
    "The capital of France is",
    "def fibonacci(n):",
    "Hugging Face is an open-source company",
    "请用一句话解释什么是 KV Cache。",
    "Translate to French: The weather is nice today, so we decided to go for a walk in the park.",
    '{"name": "mini-vllm", "version": "0.1.0", "modules": ["scheduler",',
    "Question: A train travels at 80 km/h for 2.5 hours. How far does it go?\nAnswer: Let's think step by step.",
    "春节是中国最重要的传统节日。每到春节前夕，在外工作的人们会想尽办法回到家乡，与家人团聚。在不同的地区，过年的习俗",
    "Once upon a time, in a small village at the edge of a vast forest, there lived an old clockmaker who could repair any clock in the world except",
    "import torch\nimport torch.nn as nn\n\nclass RMSNorm(nn.Module):\n    def __init__(self, dim, eps=1e-6):",
    "The history of computing hardware spans several centuries. Early mechanical calculators could perform basic arithmetic but required a human operator for every step. In the nineteenth century, Charles Babbage designed the Analytical Engine, and Ada Lovelace wrote what is often described as the first program for it. The twentieth century brought vacuum tubes, then transistors, then integrated circuits, each making computers smaller, faster, and cheaper. From the 1970s onward, the microprocessor put an entire CPU on a single chip. Looking back over this history, one recurring theme stands out:",
    "大语言模型的推理过程可以分为两个阶段。第一个阶段称为预填充，模型一次性读入全部提示词，并行计算每个位置的隐藏状态，同时把每一层的键和值保存下来。第二个阶段称为解码，模型每一步只生成一个新词，并且需要读取之前保存的全部键和值。解码阶段每一步的计算量很小，但需要从显存中搬运大量数据，因此它通常受限于显存带宽而不是算力。为了提高吞吐量，推理引擎会",
    "Here is a list of ten tips for writing clean Python code:\n1. Use meaningful variable names.\n2. Keep functions short and focused.\n3.",
    "You are a helpful assistant for an engineering team building LLM inference systems. Answer precisely and concisely, and state the likely bottleneck first.\nQuestion: Why is the decode phase usually memory-bound?\nAnswer:",
    "You are a helpful assistant for an engineering team building LLM inference systems. Answer precisely and concisely, and state the likely bottleneck first.\nQuestion: How would you test a paged KV cache for correctness?\nAnswer:",
    "You are a helpful assistant for an engineering team building LLM inference systems. Answer precisely and concisely, and state the likely bottleneck first.\nQuestion: When does CUDA Graph help?\nAnswer:",
    "<|im_start|>user\n用三句话介绍 PagedAttention 的核心思想。<|im_end|>\n<|im_start|>assistant\n",
]

ref = []

for i, p in enumerate(prompts):
    inputs = tokenizer(p, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
    new_ids = outputs[0, inputs.input_ids.shape[1]:].tolist()
    ref.append({
        "prompt": p,
        "output": tokenizer.decode(new_ids, skip_special_tokens=True),
        "input_ids": inputs.input_ids[0].tolist(),
        "new_ids": new_ids,
    })
    out_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"[{i:02d}] {p}")
    print(f"  -> {out_text}")
    print("-" * 60)

json.dump(ref, open("ref.json", "w"), ensure_ascii=False)