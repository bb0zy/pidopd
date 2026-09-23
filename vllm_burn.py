"""vLLM GPU burn test — 跑 vLLM 推理，观察 nvitop 的 util 读数是否正常"""
import time
from vllm import LLM, SamplingParams

# ===== 配置（改成你的模型路径）=====
MODEL_PATH = "/models/qwen"  # 改成你的模型
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 512
NUM_QUESTIONS = 8  # 并发多少条请求
# ===================================

print(f"模型: {MODEL_PATH}")
print(f"max_model_len: {MAX_MODEL_LEN}")
print(f"并发请求数: {NUM_QUESTIONS}")
print()

# 初始化 vLLM
print("加载 vLLM...")
start = time.time()
llm = LLM(
    model=MODEL_PATH,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=0.9,
    enable_prefix_caching=False,
    seed=42,
)
print(f"vLLM 加载完成: {time.time() - start:.1f}s")
print()

# 准备 prompt
prompts = [
    "Please solve this step by step. What is 15 * 23 + 47 - 89?"
] * NUM_QUESTIONS

sampling = SamplingParams(
    n=1,
    temperature=0.8,
    top_p=1.0,
    max_tokens=MAX_NEW_TOKENS,
)

# 跑推理，多轮，让 GPU 持续忙碌
print("开始 vLLM 推理，跑 5 轮...")
print("-" * 60)
print(">>> 现在盯 nvitop，看 util 是否正常 <<<")
print("-" * 60)

start = time.time()
for round_id in range(5):
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    t1 = time.time()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  round {round_id+1}/5 | {t1-t0:6.2f}s | {total_tokens:5d} tokens | "
          f"{total_tokens/(t1-t0):8.1f} tok/s")

elapsed = time.time() - start
print("-" * 60)
print(f"总时长: {elapsed:.1f}s")
print(f"平均吞吐: {5 * NUM_QUESTIONS * MAX_NEW_TOKENS / elapsed:.1f} tok/s")
print()
print("=" * 60)
print(">>> 现在看 nvitop 的 util 读数 <<<")
print(">>> 正常应该 70-100%，如果还是 0% 就是 vLLM 的问题 <<<")
print("=" * 60)
