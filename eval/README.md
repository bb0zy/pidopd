# 本地 LLM 数学评测

已在开发机安装 pandas/pyarrow 并检查本地数据；没有安装新 GPU 环境或运行模型评测。请在公司 Linux GPU 服务器使用。

默认使用 vLLM 推理，可切换 Transformers 单 GPU 推理；TRL 的 `apply_chat_template` 用于统一对话提示词，Transformers 加载本地 tokenizer。这里评估生成答案质量与离线批处理速度，不进行训练，也不使用 SFT loss 代替答案准确率。

## 版本与服务器环境

目标环境是 accelerate 1.15.0、torch 2.10.0+cu128、Transformers 5.3.0、TRL 1.13.0、vLLM 0.19.1；Python 沿用 3.12.3。Transformers 模型加载使用 `dtype`，TRL 模板调用使用 `trl.data_utils.apply_chat_template(..., processing_class=tokenizer)`。

版本由根目录 `constraints.txt` 统一约束，`eval/requirements.txt` 引用共有依赖，vLLM 单列在 `requirements-vllm.txt`。官方 vLLM 0.19.1 虽依赖 torch 2.10.0，但明确排除 Transformers 5.3.x；因此保留你的目标版本并不意味着这组官方 wheel 可以直接安装或运行。使用服务器实际 wheel 元数据和运行结果确认，不能以 `--no-deps` 当作兼容修复。详见根目录 README；可自行运行 `python check_environment.py --profile vllm`。脚本不自动修改包或回退后端。

官方参考：

- https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/cuda.txt
- https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/common.txt
- https://github.com/huggingface/trl/blob/v1.13.0/trl/data_utils.py

## 模型与数据

默认模型目录是 `/models/qwen`；通常使用 `--model /absolute/path/to/model` 指定完整本地模型目录，包含 config、tokenizer、权重及其分片。仅支持已合并的完整因果语言模型；不直接加载 LoRA adapter。

默认数据目录固定为项目根目录下的 `dataset/eval`，不受启动工作目录影响；逐文件评测其中 `.parquet`、`.jsonl`、`.json`，不会递归扫描其他目录。已用 pandas/pyarrow 解码检查本地 AIME24、AIME25 和 HMMT：每个文件 30 条，字段为 `prompt`、`data_source`、`ability`、`reward_model`、`extra_info`。其中 `reward_model` 包含 `ground_truth/style`，`extra_info` 仅有 `index/split`，没有独立纯问题或解题过程字段。JSON 必须为记录数组，JSONL 每行一条记录。数据检查不代表已运行模型评测；评测脚本本身不依赖 pandas。

自动字段选择：

- 题目：`prompt`、`messages`、`problem`、`question`；可以是字符串或 role/content 对话列表。
- 标准答案：`reward_model.ground_truth`、`answer`、`ground_truth`、`target`。
- 可用 `--prompt-key question --answer-key reward_model.ground_truth` 指定字段，支持嵌套点路径。

例如一条记录：

```json
{"prompt": [{"role": "user", "content": "What is 1+1?"}], "reward_model": {"ground_truth": "2"}}
```

提示必须以 user 消息结尾；不会将标准答案附加到提示。本地 90 条记录均含固定后缀 `Please reason step by step, and put your final answer within \boxed{}.`，构造提示时只删除该精确后缀，保留题目正文及数学符号；不会修改原始 Parquet 文件。然后添加 system 消息要求逐步推理，并将最后一行写成 `Answer: <答案>`。已有首条 system 消息会保留并追加这些要求；自定义数据的其他格式指令不会被猜测删除，需自行确保不冲突。聊天模板必须支持 system role；`--plain-prompt` 则把同样的指令作为文本前缀，不添加角色标记。数据为空、字段错误或超过上下文预算都会明确报错，不静默跳过或截断。

例如 AIME25 第一题构造成如下消息，再由本地 tokenizer 套用聊天模板：

```text
system:
Solve the following math problem step by step.
The last line of your response should be of the form
Answer: $Answer (without quotes) where $Answer is the answer to the problem.
Remember to put your answer on its own line after "Answer:".

user:
Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.
```

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
  --samples 32 --temperature 0.6 --top-p 0.95 \
  --batch-size 2 --max-model-len 32768 --max-new-tokens 31744
```

Transformers 单卡，同样的数据与提示处理：

```bash
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py \
  --model /models/your-model --backend transformers --batch-size 2
```

其他参数：`--data` 可以是单文件或目录；`--limit 5` 取每个文件前 5 题；`--dtype bfloat16` 为默认精度；`--plain-prompt` 适用于无 chat template 的 base model；`--chat-template-kwargs '{"enable_thinking":false}'` 传递模型模板支持的选项。默认每题独立采样 32 次、temperature=0.6、top_p=0.95；也可用 `--samples 16`，多次采样要求 temperature 大于零。采样参数可按模型调整，比较模型时应保持一致；avg@32 本身不规定具体 temperature。

默认总上下文上限为 32768 tokens（输入提示与生成之和），最大生成预算为 31744 tokens；这是本项目数学评测的起始配置，不代表所有题目都需要或足够使用该长度。代码会检查提示长度加完整生成预算，因此默认允许最多 1024 tokens 的输入提示（含聊天模板）。生成遇到结束标记会提前停止，不会强制输出满 31744 tokens。

数学题的长度需求取决于题目、模型及推理方式。评测后查看 `length_finish_rate` 和被截断的逐题输出：若长度上限频繁触发，应提高生成预算并相应调整总上下文；若提示超过默认的 1024 tokens 余量，应调整总上下文或生成预算。比较模型时应保持预算一致，并确保设置不超过模型真实上下文能力。

`--batch-size` 默认 2，控制一次提交的题数；默认每批生成 2 × 32 = 64 个 completion。vLLM 默认显存比例 0.9，可通过 `--gpu-memory-utilization` 调整。当前 Transformers 后端使用 `.to("cuda")` 单卡加载，尚未配置 `device_map="auto"`，因此模型须能装入单 GPU。请直接运行脚本，不使用 torchrun 启动重复评测进程。

`--max-num-seqs` 是 vLLM 调度并发序列上限，默认显式设置为 32，不再依赖版本/硬件默认值。它不会改变每题的采样次数，也不会限制评测题数。可设为 8 让引擎分批调度；设得更高只有在待处理序列足够、KV cache/token 预算允许时才有用。初始化后打印实际 `max_num_seqs` 和 `max_num_batched_tokens`，并写入 `config.json` 的 `vllm_scheduler`。Transformers 后端不使用这个参数。

## 输出与指标含义

每次创建独立的 `eval/results/<UTC时间>/`，保存实际参数/软件版本/GPU信息的 `config.json`、每个数据文件的逐题 `*.predictions.jsonl`，以及 `summary.json`。逐题输出包含 prompt、参考答案、完整生成文本、提取答案、正确性、token 数、结束原因。每个批次写入并 flush，失败时已有明细保留；不实现断点续跑。

- `avg@32`：主评测指标。每题独立采样 32 次，计算正确次数 / 32，再对该数据集全部题目取平均：`avg@32 = (1/N) * sum(correct_i / 32)`。输出范围为 0–1，乘以 100 得百分比。例如一题答对 16 次，该题得分为 0.5。它不采用“32 次中任意一次正确就算对”或多数投票评分。逐题明细同时保存 `correct_samples` 和该题 `avg@32`；汇总按数据文件分别报告。

默认 `--batch-size 2`，即一次提交两题、每题 32 个答案，仍需根据显存调整批量和生成长度。`--samples` 允许 16 或 32；使用 16 时，每题实际计算的是正确次数 / 16，但当前代码仍将字段命名为 `avg@32`，读结果时必须结合 `samples_per_question` 判断，不能把它当作严格的 avg@32。已移除 `--pass-k`。预热会为一题生成指定采样次数的答案，预热结果不计入评分。
- `output_tokens_per_second`：实际生成 token 数 / 计时区间总秒数；计入结束 token，不包含 Transformers 补齐 token。计时覆盖分批推理调用、解码及返回结果，排除模型初始化、预热、评分和写文件。
- `mean_batch_seconds`、`median_batch_seconds`：整批完成耗时，并非单请求延迟、TTFT 或 TPOT。本脚本不是在线并发服务压测。
- `length_finish_rate`：达到生成长度上限比例；`answer_extraction_failure_rate`：未提取出最终答案比例，均仍计入准确率分母。

默认用一题完整生成预热一次（可能耗时较长），`--warmup 0` 可关闭。vLLM 关闭前缀缓存，以减少预热对重复题目的缓存影响。随机种子固定，但不同后端/硬件/批次划分不保证逐 token 一致。比较速度请保持提示、精度、采样参数、生成长度、batch size、硬件一致；不同答案长度仍影响测量。

评分优先取最后一行匹配行首 `Answer:` / `Answer：` 的答案（大小写不敏感，也兼容 `Final answer:` / `答案：`）；若含已闭合的 `</think>`，只检查其后的回答。支持 `Answer: $70$` 以及该行内的 boxed 答案。没有显式答案行时兼容完整 `\boxed{...}` 或仅含数字的回答；绝不从推理正文随意抓最后一个数字。

`answer_grader.py` 使用 Math-Verify 的 LaTeX 解析和 SymPy 数学等价判定，处理整数、分数、根式、π、阶乘和多解。依赖 `math-verify>=0.8.0,<1.0` 已加入评测 requirements，其 SymPy/LaTeX 解析依赖由该包管理；训练不要求安装评分器。本机未安装此新依赖、未运行评分回归检查，服务器须准备依赖后验证。

- `8\sqrt{10}` 与 `\sqrt{640}` 等表达式使用数学判定，不只比较字符。
- 当参考答案明确带度数标记时，允许省略该单位，例如 AIME25 的 `336^\circ` 与 `336`；不进行角度/弧度自动换算，也不随意删除其他单位。
- 顶层逗号分隔的多个答案按无序列表一对一匹配，数量必须一致，不能只提交其中一个解。显式 `\{...\}` 也可作集合外包装；括号元组不自动当作无序根集合。逗号不自动解释为千位分隔符，请输出无千位分隔符的整数。
- 浮点判定使用 Math-Verify 的 `float_rounding=12`、`numeric_precision=30`，并非任意精度近似都接受；精确值应优先输出分数/根式。
- 每次数学解析/比较设置 5 秒超时，答案最大 2048 字符。评分在 Linux 主线程进行；复杂表达式仍可能不被解析器支持，结果不保证对任意数学语法完备。
- 参考答案在模型加载前必须成功解析，否则带文件名、行号报错。模型答案缺失、无法解析或异常时判错；Math-Verify 内部捕获的比较超时可能直接返回不等，不保证每次都提供详细超时原因。

逐条 completion 新增 `grading_method`（exact / math_verify / missing_answer / answer_count_mismatch / parse_or_verification_error），异常时附 `grading_error`；`text`、`extracted_answer` 和 `reference` 保留供复核。软件版本记录包含 math-verify、SymPy 和 LaTeX 解析库。avg@32 的计算方式不变。

可在服务器单独运行 CPU 评分检查，不加载模型：

```bash
python -m unittest discover -s eval -p test_answer_grader.py
```

检查包含数学等价/错误答案、多解缺漏和重复，以及三个本地数据集全部参考答案的可解析性。旧结果不会自动重新评分。
