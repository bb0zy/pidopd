# pidopd

数学模型的 LoRA 在线训练与离线答案评测。学生通过 Transformers 生成回答，同一个 vLLM teacher 分别使用普通提示和包含参考解答的提示评分；`MyTrainer` 继承 TRL `SFTTrainer`，使用自定义 top-K 加权目标训练。评测支持原始模型、最终 LoRA 和任意中间 checkpoint。

所有命令从项目根目录执行，训练和模型评测面向 Linux / NVIDIA GPU 服务器。

## 项目结构

```text
requirements.txt             # 统一的训练、评测依赖
train/
  main.py                    # 数据加载、模型和 LoRA 配置、启动训练
  myconfig.py                # Trainer 扩展配置
  mytrainer.py               # 学生生成、自定义 loss、训练步骤
  utils.py                   # 消息准备、tokenization、padding
  vllm_client.py             # teacher 服务客户端
  outputs/<model_name>/      # 最终 LoRA 与 tokenizer
  checkpoint/<model_name>/   # checkpoint-<global_step>/
vllm/
  start_vllm.sh              # 独立启动 teacher 服务
eval/
  run_eval.py                # 原始模型 / LoRA 评测入口
  answer_grader.py           # 数学答案等价判定
  test_checkpoint_loading.py # 不加载 GPU 的模型加载路由测试
  test_answer_grader.py      # 答案评分回归测试
  results/<UTC时间>/          # 新评测结果
  result/                    # 保留的历史结果
dataset/
  train/                     # 两个训练 Parquet 分片
  eval/                      # AIME24、AIME25、HMMT
tmp/datasets/                # Datasets 本地缓存
```

## 环境与依赖

统一使用根目录 [requirements.txt](requirements.txt)，其中直接固定 Torch、Transformers、TRL、Accelerate 和 PEFT 版本，并包含 Datasets、OpenAI 客户端、W&B、PyArrow 和 Math-Verify。目标 Python 为 3.12.3，Torch 为 `2.10.0+cu128`；安装源需要提供相应 CUDA wheel。

```bash
python -m pip install -r requirements.txt
python -m pip check
```

vLLM 使用服务器现有的、兼容 Transformers 5.3.0 的定制版 0.19.1，单独准备该构建。统一依赖文件用注释记录这一要求，不自动安装公开版 vLLM：公开版的依赖声明排除了 Transformers 5.3.x，直接加入安装清单会产生冲突。[vLLM 0.19.1 依赖声明](https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/common.txt)

已有可运行的服务器环境无需重复安装。`pip check` 检查声明的依赖关系，不能替代 CUDA 和实际模型运行验证。训练通过 `local_files_only=True` 读取模型；评测额外设置 Hugging Face 离线环境变量，不自动下载模型或数据。

## 训练

### 配置与数据

在 [train/main.py](train/main.py) 中修改 `model_path`、两个 system prompt、`LoraConfig(...)` 和 `MyConfig(...)`。teacher 服务地址与模型别名的默认值位于 [train/myconfig.py](train/myconfig.py)，也可在 `MyConfig(...)` 中覆盖。训练入口当前不解析命令行参数。

默认读取：

```text
dataset/train/train-00000-of-00002.parquet
dataset/train/train-00001-of-00002.parquet
```

原始字段使用 `problem`、`solution`、`Answer`。`prepare_messages` 生成包含 student system prompt 和问题的 `message`，保留 `stu_sys_prompt`、`teacher_sys_prompt` 和 `Answer`；teacher system prompt 自动追加 `solution`。参考解答只进入特权 teacher 提示，`Answer` 当前不参与 loss。

collator 使用同一个 tokenizer 构造两路提示并左侧 padding。`max_length=1024` 只截断 student prompt；teacher prompt 保留完整参考解答，尚未限制长度。student 与 teacher 必须有一致的 token-to-ID 映射和兼容的输出词表。

当前配置：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `per_device_train_batch_size` | 2 | 每设备每个 micro-batch 的题数 |
| `gradient_accumulation_steps` | 8 | 每8个 micro-batch 更新一次参数 |
| `learning_rate` | 1e-5 | 学习率 |
| `num_train_epochs` | 1 | 遍历数据的轮数 |
| `max_new_tokens` | 2048 | 每次学生回答的生成上限 |
| `loss_top_k` | 16 | 每个有效回答位置参与 loss 的候选数 |
| `save_steps` | 50 | 每50个优化器更新步保存 checkpoint |
| `logging_steps` | 10 | 日志间隔 |
| `eval_strategy` | `no` | 训练内评估关闭，使用独立评测入口 |
| LoRA | r=16，alpha=32，all-linear | 当前 adapter 配置 |

### 启动 teacher 与学生

先在一个终端启动 teacher：

```bash
VLLM_GPU=1 bash vllm/start_vllm.sh /models/teacher
```

再在另一个终端启动学生：

```bash
wandb login
CUDA_VISIBLE_DEVICES=0 python -m train.main
```

当前 `report_to="wandb"`，在线记录需要登录；需要离线记录时可设置 `WANDB_MODE=offline`。项目名可通过 `WANDB_PROJECT` 设置，运行名在 `main.py` 的 `run_name` 中设置。

teacher 启动脚本默认使用 GPU 1、地址 `127.0.0.1`、端口8001、模型别名 `teacher`、上下文上限10240、最大并发8、显存比例0.8、tensor parallel size=1。客户端默认连接 `http://127.0.0.1:8001/v1`。

可通过 `VLLM_GPU`、`VLLM_HOST`、`VLLM_PORT`、`VLLM_MODEL_NAME`、`VLLM_MAX_MODEL_LEN`、`VLLM_MAX_NUM_SEQS`、`VLLM_GPU_MEMORY_UTILIZATION`、`VLLM_TENSOR_PARALLEL_SIZE` 和 `VLLM_DTYPE` 覆盖对应设置；额外 vLLM 参数直接接在模型路径后。修改服务地址或别名时同步修改客户端配置。单卡机器应明确规划 teacher 和 student 的显存分配。

### 训练目标与资源开销

每个 batch 先由学生生成回答，再用同一个 teacher 对普通上下文和特权上下文分别打分。只在有效回答位置计算：

```text
indices = topk(student_logits, K).indices
r_1 = teacher_logprob[indices] - normal_teacher_logprob[indices]
r2 = softmax(student_topk_logits.detach()) * r_1
loss = mean_over_answer_tokens(-sum_over_K(r2 * student_topk_logprob))
```

`student_topk_logprob` 来自全词表 `log_softmax`，保留梯度；teacher 差值和权重不保留梯度。两路 teacher 分数在 CPU 上按 student top-K 索引筛选并相减，只有 `[有效回答 token 数, K]` 的 `r_1` 搬到 GPU。这是实验性目标，loss 可以为负。

学生生成开启 KV cache，`unwrap_model_for_generation` 临时关闭梯度检查点，生成结束后恢复。计算 loss 的学生前向显式关闭 KV cache。teacher 由独立服务评分，不随学生更新参数。

teacher 的 `score()` 仍通过 HTTP 返回整个输入序列的全词表 logprob，JSON 解析、传输和 CPU 内存可能成为瓶颈。teacher 完整提示加学生回答必须留在服务上下文预算内；当前没有自动过滤超长 teacher 输入。正式训练前可先用较小 batch 和较短生成预算验证一个完整更新步骤。

### 保存与模型命名

`model_name` 默认取 `model_path` 的最后一级目录名。例如 `/models/qwen` 保存为：

```text
train/outputs/qwen/                         # 最终 adapter
train/checkpoint/qwen/checkpoint-50/        # 中间 checkpoint
train/checkpoint/qwen/checkpoint-100/
```

最终目录由 `trainer.save_model(final_model_dir)` 写入。LoRA 目录通常包含 `adapter_config.json`、`adapter_model.safetensors` 和 tokenizer，不包含完整基座权重；checkpoint 还保存恢复训练所需的状态。当前没有自动恢复训练逻辑。

不同模型名称分别保存。同一模型的不同实验若需分别保留，可显式设置不同的 `model_name`。调整保存频率使用 `save_steps`，限制 checkpoint 数量可设置 `save_total_limit`。

## 评测

### 原始模型、最终 LoRA 与 checkpoint

```bash
# Base model or a complete merged model
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --model /models/qwen

# Final LoRA adapter
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py \
  --model /models/qwen --lora-path train/outputs/qwen

# An intermediate checkpoint
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py \
  --model /models/qwen --lora-path train/checkpoint/qwen/checkpoint-50

# Transformers backend, single GPU
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py \
  --model /models/qwen --lora-path train/checkpoint/qwen/checkpoint-50 \
  --backend transformers

# vLLM tensor parallelism
CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval.py \
  --model /models/qwen --tensor-parallel-size 2
```

`--model` 始终指向完整的本地基座模型，必须与 adapter 训练时的权重一致；`--lora-path` 指向最终目录或任意 checkpoint。不传 `--lora-path` 时评测原始模型。vLLM 使用 `LoRARequest`，Transformers 使用 PEFT，无需提前合并模型。优先使用 adapter 目录保存的 tokenizer，没有 `tokenizer_config.json` 时使用基座 tokenizer。这一路径面向当前未扩充词表的普通 LoRA 配置。

Transformers 后端将模型加载到单张 GPU，尚未使用 `device_map="auto"`；vLLM 支持 tensor parallelism。直接运行入口，不使用 torchrun 启动重复评测进程。

### 数据与提示

默认逐文件评测 `dataset/eval` 下的 AIME24、AIME25、HMMT，各30题。`--data` 可指定单文件或目录，支持 Parquet、JSON数组和 JSONL，不递归扫描。`--limit 5` 只取每个文件前5题。

题目默认依次查找 `prompt`、`messages`、`problem`、`question`，答案依次查找 `reward_model.ground_truth`、`answer`、`ground_truth`、`target`。可用 `--prompt-key` 和 `--answer-key` 指定字段，支持点分隔的嵌套路径。

提示必须以 user 消息结尾，不包含参考 assistant 解答。评测删除题目末尾精确匹配的旧指令 `Please reason step by step, and put your final answer within \boxed{}.`，添加要求逐步推理、最后一行输出 `Answer: <答案>` 的 system 提示。已有 system 消息会追加这段要求；原始数据文件不会被修改。

`--plain-prompt` 适合无 chat template 的 base model，将相同指令作为文本前缀；默认使用 TRL 对话模板。可用 `--chat-template-kwargs '{"enable_thinking":false}'` 传递模型模板支持的选项。默认不运行自定义模型代码，需要时显式传入 `--trust-remote-code`。

### 生成参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--backend` | vllm | 可选 transformers |
| `--samples` | 32 | 每题采样数，可选16或32 |
| `--temperature` / `--top-p` | 0.6 / 0.95 | 采样参数，多次采样要求 temperature > 0 |
| `--batch-size` | 2 | 每次提交题数，默认每批共64个 completion |
| `--max-num-seqs` | 32 | vLLM 并发序列上限，不改变每题采样数 |
| `--max-model-len` | 32768 | 提示加回答的总长度上限 |
| `--max-new-tokens` | 31744 | 每个回答生成上限 |
| `--dtype` | bfloat16 | 推理精度 |
| `--gpu-memory-utilization` | 0.9 | vLLM 显存比例 |
| `--warmup` | 1 | 不计时的预热次数；0表示关闭 |
| `--seed` | 42 | 随机种子 |

默认长度设置留给提示1024个 token（含模板）；超过预算会报错，不会自动截断。根据模型实际上下文能力调整长度，评测后查看 `length_finish_rate`。预热同样使用完整生成预算，可能耗时较长。比较原始模型与不同 checkpoint 时，保持数据、提示、预算、后端和采样设置一致；不同硬件和批次划分不保证逐 token 一致。

### 输出与评分

每次创建独立目录 `eval/results/<UTC时间>/`，可通过 `--output` 修改父目录：

- `config.json`：实际参数、基座与 adapter 路径、tokenizer 路径、包版本、GPU、vLLM 调度配置。
- `*.predictions.jsonl`：逐题提示、参考答案、所有生成文本、提取答案、正确性、token数、结束原因与评分方法；逐批写入，不支持断点续跑。
- `summary.json`：按数据集汇总准确率、生成时间、吞吐、截断率和答案提取失败率。

`avg@32` 是每题正确采样数除以32，再对题目取平均，不是 pass@32 或多数投票。`--samples 16` 时实际分母为16，但当前输出字段仍名为 `avg@32`，需结合 `samples_per_question` 判断。

计时包括生成、解码和返回结果，排除模型初始化、预热、评分和写文件。`output_tokens_per_second` 是输出 token 总数除以生成计时；批次耗时不是 TTFT 或在线请求延迟。vLLM 关闭前缀缓存。生成遇到 EOS 会提前结束，截断和未提取到答案的样本仍计入准确率分母。

答案提取优先使用最后一个行首 `Answer:`，兼容 `Final answer:`、`答案：`、完整 `\boxed{...}` 和纯数字回答；有已闭合 `</think>` 时只检查其后的文本，不从推理过程随意取数字。

评分使用 Math-Verify 和 SymPy 判定数学等价，支持分数、根式、π、阶乘和多个解。顶层逗号分隔的多个解按无序列表一对一匹配，缺解或重复解不算正确；括号元组不自动视为无序集合。参考答案带角度标记时允许省略单位，不进行角度/弧度转换。每次解析/比较超时5秒，答案最大2048字符。参考答案不可解析会在加载模型前报错；模型答案缺失、不可解析或异常判错，并记录评分方法或错误信息。

## 检查

无需模型和 GPU 的加载路由测试，覆盖原始模型、最终 adapter、中间 checkpoint、tokenizer 回退和 vLLM rank 容量：

```bash
python -B -m unittest discover -s eval -p test_checkpoint_loading.py -v
```

在安装评分依赖的 Linux 环境检查数学判定与本地数据集参考答案：

```bash
python -B -m unittest discover -s eval -p test_answer_grader.py -v
```

加载路由测试使用模拟后端，不能替代实际 GPU 推理。训练内评估链路仍未适配，保持 `eval_strategy="no"`，通过独立评测入口比较 checkpoint。
