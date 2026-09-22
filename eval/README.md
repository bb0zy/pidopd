# 本地 LLM 数学评测

仅编写代码，未在开发机安装、导入、运行或测试任何评测依赖。请在公司 Linux GPU 服务器使用。

默认使用 vLLM 推理，可切换 Transformers 单 GPU 推理；TRL 的 `apply_chat_template` 用于统一对话提示词，Transformers 加载本地 tokenizer。这里评估生成答案质量与离线批处理速度，不进行训练，也不使用 SFT loss 代替答案准确率。

## 版本与服务器环境

目标环境是 Python 3.12.3、CUDA 12.9、torch 2.12+cu129、vLLM 0.23.0。
**这组版本不能宣称官方 wheel 直接兼容**：vLLM 0.23.0 的官方 `requirements/cuda.txt` 指定 `torch==2.11.0`，并包含 CUDA 13 相关依赖。保留 torch 2.12+cu129 时，需要公司已有的兼容 vLLM 构建，或依据官方文档针对现有 PyTorch/CUDA 构建、适配及验证。不能仅使用 `--no-deps` 绕过二进制 ABI 冲突。

`requirements.txt` 仅列出应用层 API 版本范围，刻意不安装/替换 torch 或 vLLM；它不是经过运行验证的锁文件。服务器准备兼容基础环境后，再解析这些依赖并确认安装计划不替换基础组件。离线安装需事先准备完整且兼容的 wheel 集合；本项目不提供或下载这些包。新模型若需要 Transformers 5，请另行调整 TRL/Transformers 组合，当前范围不保证支持。

官方参考：

- https://github.com/vllm-project/vllm/blob/v0.23.0/requirements/cuda.txt
- https://docs.vllm.ai/en/v0.23.0/getting_started/installation/gpu/
- https://huggingface.co/docs/trl/v0.23.0/data_utils

## 模型与数据

默认模型目录是项目根目录 `models/model`；通常使用 `--model /absolute/path/to/model` 指定完整本地模型目录，包含 config、tokenizer、权重及其分片。仅支持已合并的完整因果语言模型；不直接加载 LoRA adapter。

默认数据目录固定为项目根目录下的 `dataset/eval`，不受启动工作目录影响；逐文件评测其中 `.parquet`、`.jsonl`、`.json`，不会递归扫描其他目录。已针对当前 AIME/HMMT 文件可见的 `prompt`、`reward_model.ground_truth` 结构编写适配，但未执行 Parquet 解码验证。JSON 必须为记录数组，JSONL 每行一条记录。

自动字段选择：

- 题目：`prompt`、`messages`、`problem`、`question`；可以是字符串或 role/content 对话列表。
- 标准答案：`reward_model.ground_truth`、`answer`、`ground_truth`、`target`。
- 可用 `--prompt-key question --answer-key reward_model.ground_truth` 指定字段，支持嵌套点路径。

例如一条记录：

```json
{"prompt": [{"role": "user", "content": "What is 1+1?"}], "reward_model": {"ground_truth": "2"}}
```

提示必须以 user 消息结尾；不会将标准答案附加到提示。缺少 boxed 指令时会追加最终答案格式要求。数据为空、字段错误或超过上下文预算都会明确报错，不静默跳过或截断。

脚本在导入库前强制设置 Hugging Face 离线模式，模型路径必须为存在的本地目录；数据通过本地文件读取，不调用 Hub 下载接口。默认禁止自定义模型代码；只有确有需要时使用 `--trust-remote-code`，该选项允许执行本地模型目录中的代码，其额外行为由该代码决定。

## 运行方式（由你在服务器执行）

在项目根目录执行默认 vLLM 评测：

```bash
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --model /models/your-model
```

两卡 tensor parallel，每题采样 32 次计算 avg@32：

```bash
CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval.py \
  --model /models/your-model --tensor-parallel-size 2 \
  --samples 32 --temperature 0.6 --top-p 1.0 \
  --batch-size 1 --max-model-len 8192 --max-new-tokens 4096
```

Transformers 单卡，同样的数据与提示处理：

```bash
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py \
  --model /models/your-model --backend transformers --batch-size 1
```

其他参数：`--data` 可以是单文件或目录；`--limit 5` 取每个文件前 5 题；`--dtype bfloat16` 为默认精度；`--plain-prompt` 适用于无 chat template 的 base model；`--chat-template-kwargs '{"enable_thinking":false}'` 传递模型模板支持的选项。每题固定独立采样 32 次，默认 temperature=0.6、top_p=1.0，temperature 必须大于零。采样参数可按模型调整，比较模型时应保持一致；avg@32 本身不规定具体 temperature。

默认总上下文上限为 8192 tokens（输入提示与生成之和），最大生成预算为 4096 tokens；这是本项目数学评测的起始配置，不代表所有题目都需要或足够使用该长度。代码会检查提示长度加完整生成预算，因此默认允许最多 4096 tokens 的输入提示（含聊天模板）。生成遇到结束标记会提前停止，不会强制输出满 4096 tokens。

数学题的长度需求取决于题目、模型及推理方式，复杂 AIME/HMMT 题的长推理可能超过 4096 tokens。评测后查看 `length_finish_rate` 和被截断的逐题输出：若长度上限频繁触发，应提高生成预算并相应调整总上下文；比较模型时应保持预算一致。还需确保设置不超过模型真实上下文能力。

`--batch-size` 控制一次提交的题数，实际 completion 数还需乘以 samples。vLLM 默认显存比例 0.9，可通过 `--gpu-memory-utilization` 调整。当前 Transformers 后端使用 `.to("cuda")` 单卡加载，尚未配置 `device_map="auto"`，因此模型须能装入单 GPU。请直接运行脚本，不使用 torchrun 启动重复评测进程。

## 输出与指标含义

每次创建独立的 `eval/results/<UTC时间>/`，保存实际参数/软件版本/GPU信息的 `config.json`、每个数据文件的逐题 `*.predictions.jsonl`，以及 `summary.json`。逐题输出包含 prompt、参考答案、完整生成文本、提取答案、正确性、token 数、结束原因。每个批次写入并 flush，失败时已有明细保留；不实现断点续跑。

- `avg@32`：主评测指标。每题独立采样 32 次，计算正确次数 / 32，再对该数据集全部题目取平均：`avg@32 = (1/N) * sum(correct_i / 32)`。输出范围为 0–1，乘以 100 得百分比。例如一题答对 16 次，该题得分为 0.5。它不采用“32 次中任意一次正确就算对”或多数投票评分。逐题明细同时保存 `correct_samples` 和该题 `avg@32`；汇总按数据文件分别报告。

默认 `--batch-size 1`，即一次提交一题的 32 个答案，仍需根据显存调整生成长度。`--samples` 仅允许 32，避免把其他采样次数标为 avg@32；已移除 `--pass-k`。预热也会为一题生成 32 个答案，预热结果不计入评分。
- `output_tokens_per_second`：实际生成 token 数 / 计时区间总秒数；计入结束 token，不包含 Transformers 补齐 token。计时覆盖分批推理调用、解码及返回结果，排除模型初始化、预热、评分和写文件。
- `mean_batch_seconds`、`median_batch_seconds`：整批完成耗时，并非单请求延迟、TTFT 或 TPOT。本脚本不是在线并发服务压测。
- `length_finish_rate`：达到生成长度上限比例；`answer_extraction_failure_rate`：未提取出最终答案比例，均仍计入准确率分母。

默认用一题完整生成预热一次（可能耗时较长），`--warmup 0` 可关闭。vLLM 关闭前缀缓存，以减少预热对重复题目的缓存影响。随机种子固定，但不同后端/硬件/批次划分不保证逐 token 一致。比较速度请保持提示、精度、采样参数、生成长度、batch size、硬件一致；不同答案长度仍影响测量。

评分取最后一个完整 `\boxed{...}`，或明确的 `Final answer:` / `答案：`，或仅含数字的回答；绝不从推理正文随意抓最后一个数字。整数、小数、简单分数按精确有理数比较，因此 `023` 与 `23` 一致。其他答案仅做有限格式归一化后的精确匹配，**不提供通用符号代数等价判定**，多解、表达式及复杂 LaTeX 需按任务扩展评分器。未提取出的输出一律判错。
