# pidopd

数学模型的 LoRA 在线训练与离线答案评测。学生通过 vLLM 批量生成回答，Transformers 负责带梯度的前向与训练；同一个冻结的 Transformers teacher 分别使用普通提示和包含参考解答的提示评分。`MyTrainer` 继承 TRL `SFTTrainer`，使用自定义 top-K 加权目标训练。评测支持原始模型、最终 LoRA 和任意中间 checkpoint，可使用 vLLM 或 Transformers 后端。

所有命令从项目根目录执行，训练和模型评测面向 Linux / NVIDIA GPU 服务器。

## 项目结构

```text
requirements.txt             # 统一的训练、评测依赖
train/
  main.py                    # 数据加载、模型和 LoRA 配置、启动训练
  myconfig.py                # Trainer 扩展配置
  mytrainer.py               # 学生生成、自定义 loss、训练步骤
  test_loss_regularization.py # KL 数值测试
  student_generation.py      # 跨卡 vLLM 生成、实时权重同步、回答 padding
  test_student_generation.py # 权重同步顺序与回答对齐测试
  utils.py                   # 消息准备、tokenization、padding
  teacher_client.py          # Transformers 教师子进程与 GPU 候选评分
  test_teacher_client.py     # 分块评分对齐与通信测试
  outputs/<model_name>/      # 最终 LoRA 与 tokenizer
  checkpoint/<model_name>/   # checkpoint-<global_step>/
vllm/
  start_vllm.sh              # 手动启动 GPU1 学生生成服务
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

统一使用根目录 [requirements.txt](requirements.txt)，其中直接固定 Torch、Transformers、TRL、Accelerate 和 PEFT 版本，并包含 Datasets、NumPy、W&B、PyArrow 和 Math-Verify。目标 Python 为 3.12.3，Torch 为 `2.10.0+cu128`；安装源需要提供相应 CUDA wheel。

```bash
python -m pip install -r requirements.txt
python -m pip check
```

学生训练生成需要 vLLM，教师评分仍只用 Transformers；评测的 vLLM 后端也保留。使用服务器现有的、兼容 Transformers 5.3.0 的定制版 vLLM 0.19.1。统一依赖文件用注释记录这一要求，不自动安装公开版 vLLM：公开版的依赖声明排除了 Transformers 5.3.x，直接加入安装清单会产生冲突。[vLLM 0.19.1 依赖声明](https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/common.txt)

已有可运行的服务器环境无需重复安装。`pip check` 检查声明的依赖关系，不能替代 CUDA 和实际模型运行验证。训练通过 `local_files_only=True` 读取模型；评测额外设置 Hugging Face 离线环境变量，不自动下载模型或数据。

## 训练

### 配置与数据

学生与教师模型路径的默认值分别在 [train/myconfig.py](train/myconfig.py) 的 `student_model_path`、`teacher_model_path` 中设置。命令行 `--student_model_path`、`--teacher_model_path` 可分别覆盖默认值，也支持连字符写法 `--student-model-path`、`--teacher-model-path`；修改学生路径不会改变教师路径。两个 system prompt、`LoraConfig(...)` 和其他训练参数在 [train/main.py](train/main.py) 中修改。学生 vLLM 的卡号、模型路径和显存预算改在 `vllm/start_vllm.sh` 中设置；教师仍由训练脚本自动启动，`teacher_gpu="1"`。

训练与评测使用相同的 student system prompt，仅要求逐步解题，并在最后一行用 `\boxed{...}` 包裹最终答案，不额外要求反思或检查。后续修改 `stu_sys_prompt` 时，同步修改 `eval/run_eval.py` 中的 `SYSTEM_PROMPT`。teacher 在相同提示后追加参考解答的使用说明。

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
| `per_device_train_batch_size` | 32 | 每次生成与带梯度前向都处理32题 |
| `gradient_accumulation_steps` | 1 | 每个 batch 更新一次参数，单卡有效 batch 为32 |
| `student_vllm_server_url` | `http://127.0.0.1:8000` | 训练连接的学生生成服务 |
| `student_vllm_group_port` | 51216 | NCCL 权重同步端口，与 HTTP 端口不同 |
| 脚本 `GPU_MEMORY_UTILIZATION` | 0.5 | 常驻 vLLM 的显存预算占 GPU1 总显存的比例 |
| `learning_rate` | 1e-5 | 学习率 |
| `num_train_epochs` | 1 | 遍历数据的轮数 |
| `max_new_tokens` | 2048 | 每次学生回答的生成上限 |
| `loss_top_k` | 16 | 每个有效回答位置参与 loss 的候选数 |
| `kl_beta` | 1.0 | 相对初始学生基座的 KL 约束系数，设0可关闭 |
| `kl_top_k` | 64 | KL 单独保留的学生候选数，其余词合并为一类 |
| `save_steps` | 50 | 每50个优化器更新步保存 checkpoint |
| `logging_steps` | 10 | 日志间隔 |
| `eval_strategy` | `no` | 训练内评估关闭，使用独立评测入口 |
| LoRA | r=16，alpha=32，all-linear | 当前 adapter 配置 |

### 启动 teacher 与学生

GPU0 仅运行学生训练，GPU1 同时放置学生 vLLM 和冻结的 Transformers 教师。先在一个终端启动学生 vLLM 服务，等待服务就绪：

```bash
CUDA_VISIBLE_DEVICES=1 MODEL_PATH=/models/qwen3-1.7b bash vllm/start_vllm.sh
```

然后在另一个终端运行训练，学生模型路径必须与服务一致（两个终端都先激活服务器上的训练环境）：

```bash
wandb login
CUDA_VISIBLE_DEVICES=0 python -m train.main \
  --student_model_path /models/qwen3-1.7b
```

教师模型路径仍可用 `--teacher_model_path` 覆盖。学生服务默认 HTTP 端口8000；若脚本设置 `PORT=8001`，训练时传 `--student_vllm_server_url http://127.0.0.1:8001`。NCCL 通信端口默认51216，在 `myconfig.py` 的 `student_vllm_group_port` 中修改，由训练端发给服务建立通信组。训练不再启动或终止学生服务，退出时仅关闭自己的 NCCL 连接和教师子进程；服务需要在其终端用 Ctrl+C 关闭。

当前 `report_to="wandb"`，在线记录需要登录；需要离线记录时可设置 `WANDB_MODE=offline`。项目名可通过 `WANDB_PROJECT` 设置，运行名在 `main.py` 的 `run_name` 中设置。

训练进程只暴露 GPU0；教师子进程和学生 vLLM 服务分别设置自己的 `CUDA_VISIBLE_DEVICES=1`。改变训练卡号时修改训练命令；改变推理卡号时修改服务启动命令的 `CUDA_VISIBLE_DEVICES` 和训练配置的 `teacher_gpu`，它们都是机器或容器可见的卡号，不是父进程重映射后的编号。当前训练面向单个学生进程，不使用 `torchrun`，也不要把两张卡都暴露给 Trainer。

学生生成使用 TRL 1.13.0 的 `VLLMGeneration(mode="server")`。脚本默认只监听 `127.0.0.1`，`tensor_parallel_size=1`、`MAX_NUM_SEQS=32`、`MAX_MODEL_LEN=3072`。调整训练 batch 或生成长度时，同时调整脚本中的并发数和上下文预算。vLLM 直接接收去除 padding 后的 prompt IDs，每题生成一个回答，仅返回 token IDs，不请求 logprob 或解码文本。HTTP 只传提示与回答 IDs 等小数据；完整模型权重通过 NCCL 传输，不转换为 JSON。[TRL 生成与同步实现](https://github.com/huggingface/trl/blob/v1.13.0/trl/generation/vllm_generation.py)、[NCCL 客户端](https://github.com/huggingface/trl/blob/v1.13.0/trl/generation/vllm_client.py)

每次生成前都调用 `sync_weights()`：把当前学生的 LoRA 临时合并，经 NCCL 从 GPU0 同步完整权重到 GPU1，再恢复学生的未合并训练状态，并清空旧前缀缓存。首次生成、每次优化器更新之后的下一次生成以及恢复训练后的首次生成都经过这一同步。不会从磁盘重新加载旧 adapter；同步完成后才提交生成请求。vLLM 关闭睡眠，权重和 KV cache 内存池常驻 GPU1，避免每轮释放、重新分配。跨卡链路是否有 NVLink 会影响同步成本。

`GPU_MEMORY_UTILIZATION=0.5` 在启动脚本中调整，也可通过同名环境变量覆盖。它是 vLLM 的显存预算，不是 GPU 计算利用率目标；A100 80G 上约为一半总显存，其余留给同卡教师评分和冻结的初始学生基座模型。当前顺序是 GPU1 生成 → GPU0 学生前向并选 top-K → GPU1 教师与基座评分 → GPU0 反向传播与参数更新；没有跨 batch 流水线。当前仍让32条一起计算 loss，没有分组或隐藏的梯度累积；GPU0 不再放置 vLLM，但长回答下的全词表 logits 和计算图仍可能超过80G。

`/reset_prefix_cache` 在部分 vLLM 版本中成功后返回空响应。项目只对当前客户端的这个接口做兼容：发送真实清缓存请求并检查 HTTP 状态，不强制解析 JSON；生成与权重传输接口仍沿用 TRL 的原有处理，不修改已安装包。更换启动方式本身不能修复这个响应格式问题。

教师当前 `teacher_max_model_len=8192`、`teacher_score_chunk_size=8192`，在上下文预算内相当于整段评分。按学生最多生成2048个 token 计算，教师提示（含参考解答）可使用最多6144个 token；教师只评分，不额外生成 token。超过总长度预算时会报错，不自动截断。教师模型也需要支持该长度。

教师仍逐条处理 batch 中的输入。同一进程额外加载一个冻结的初始学生基座，用于 KL 参照；默认1.7B BF16 权重约占3.4 GB GPU1显存。`teacher_score_chunk_size` 可调小以限制单次前向的 logits 显存，用 KV cache 保留前文；它不改变上下文长度、候选数或 loss。推理进程没有固定显存比例或预分配 KV cache 配额，显存由两份模型权重、实际长度的 KV cache 和当前分块 logits 决定。

### 训练目标与资源开销

每个 batch 先由学生生成回答，再进行带梯度的学生前向，得到每个有效回答位置的 top-K 候选 ID。随后同一个 teacher 对普通上下文和特权上下文分别打分。只在有效回答位置计算：

```text
indices = topk(student_logits, K).indices
r_1 = teacher_logprob[indices] - normal_teacher_logprob[indices]
weights = softmax(student_topk_logits.detach())
r2 = weights * r_1
teacher_loss = mean_over_answer_tokens(-sum_over_K(r2 * student_topk_logprob))
loss = teacher_loss + kl_beta * kl_to_base
```

`student_topk_logprob` 来自全词表 `log_softmax`，保留梯度；teacher 差值和权重不保留梯度。教师差值按原目标直接与学生 top-K 权重相乘，不做去均值处理。教师在 GPU1 上用 `selected_logits - logsumexp(full_vocab_logits)` 得到候选 logprob，全词表归一化保持不变。只把 `[有效回答 token 数, K]` 的 float32 数组通过本地 Pipe 返回。这是实验性目标，`teacher_loss` 可以为负。

`kl_to_base` 使用同一回答上下文，比较当前学生与冻结的初始学生基座。为控制显存和跨进程传输，对学生概率最高的64个词分别计算概率，其余词合并为一个剩余概率桶，然后计算这个分组分布的 KL；这是全词表 KL 的下界，不能捕捉剩余桶内部的变化。基座在 GPU1 的教师子进程中评分，只返回 `[有效回答 token 数, 64]` 的 logprob；学生端仍以全词表归一化，不会把全词表从 GPU1 搬到 GPU0。这个约束增加第三路评分和基座模型显存，不保证单靠它就能恢复准确率。默认 `kl_beta=1.0` 是重新试跑的起点；日志记录 `teacher_loss` 与 `kl_to_base`，应结合评测结果调整系数。设 `kl_beta=0` 时不加载冻结基座，也不执行第三路评分。

训练命令可用 `--kl_beta 0.5` 或 `--kl_top_k 128` 覆盖默认值；`--kl_beta 0` 可关闭 KL 基座评分用于对照。改变 `kl_top_k` 只影响评分候选数，不改变学生生成并发和 teacher 目标的 `loss_top_k`。

学生 vLLM 生成不保留梯度，生成后根据返回长度补齐回答、构造 mask 和 labels；真实 EOS 参与 loss，回答 padding 和 prompt 不参与。计算 loss 的 Transformers 学生前向显式关闭 KV cache，继续由 Trainer 执行反向传播与优化器更新。teacher 与 KL 基座模型均保持冻结，不随学生更新参数。GPU1 评分期间，学生前向的计算图仍保留在 GPU0 上。

教师使用 `eval()`、冻结参数和 `torch.inference_mode()`，直接进行分块前向，不调用 `generate()`。每块 logits 在 GPU 上提取候选后释放，避免构造完整序列的全词表分数张量，也不把全词表搬到 CPU 或展开成 Python 对象。分块前向与整段前向可能有浮点数值差异，实际 GPU 速度仍需上机验证。没有新增计时代码。

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
python eval/run_eval.py \
  --model /models/qwen3-1.7b --tensor-parallel-size 1 \
  --lora-path train/checkpoint/qwen3-1.7b/checkpoint-50

```

`--model` 始终指向完整的本地基座模型，必须与 adapter 训练时的权重一致；`--lora-path` 指向最终目录或任意 checkpoint。不传 `--lora-path` 时评测原始模型。vLLM 使用 `LoRARequest`，Transformers 使用 PEFT，无需提前合并模型。优先使用 adapter 目录保存的 tokenizer，没有 `tokenizer_config.json` 时使用基座 tokenizer。这一路径面向当前未扩充词表的普通 LoRA 配置。

Transformers 后端将模型加载到单张 GPU，尚未使用 `device_map="auto"`；vLLM 支持 tensor parallelism。直接运行入口，不使用 torchrun 启动重复评测进程。

### 数据与提示

默认逐文件评测 `dataset/eval` 下的 AIME24、AIME25、HMMT，各30题。`--data` 可指定单文件或目录，支持 Parquet、JSON数组和 JSONL，不递归扫描。`--limit 5` 只取每个文件前5题。

题目默认依次查找 `prompt`、`messages`、`problem`、`question`，答案依次查找 `reward_model.ground_truth`、`answer`、`ground_truth`、`target`。可用 `--prompt-key` 和 `--answer-key` 指定字段，支持点分隔的嵌套路径。

提示必须以 user 消息结尾，不包含参考 assistant 解答。评测删除题目末尾精确匹配的重复指令 `Please reason step by step, and put your final answer within \boxed{}.`，添加与训练相同的 system 提示，要求逐步推理、最后一行仅输出 `\boxed{最终答案}`。已有首条 system 消息会替换为这段统一提示；原始数据文件不会被修改。

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

答案提取优先使用最后一个完整的 `\boxed{...}`，支持嵌套花括号；没有 boxed 答案时兼容旧的 `Answer:`、`Final answer:`、`答案：` 和纯数字回答。有已闭合 `</think>` 时只检查其后的文本，不从推理过程随意取数字。

评分使用 Math-Verify 和 SymPy 判定数学等价，支持分数、根式、π、阶乘和多个解。顶层逗号分隔的多个解按无序列表一对一匹配，缺解或重复解不算正确；括号元组不自动视为无序集合。参考答案带角度标记时允许省略单位，不进行角度/弧度转换。每次解析/比较超时5秒，答案最大2048字符。参考答案不可解析会在加载模型前报错；模型答案缺失、不可解析或异常判错，并记录评分方法或错误信息。

## 检查

学生生成测试覆盖首次/更新后/恢复后的同步顺序、同步失败时禁止生成、服务地址配置和 prompt/回答/EOS/padding 对齐，以及清缓存接口的空响应和 HTTP 错误处理。安装 Torch、TRL、PEFT 时还会用 CPU 小模型检查 TRL 的 LoRA 合并同步和恢复；不启动真实 vLLM，也不下载模型：

```bash
python -B -m unittest train.test_student_generation -v
```

教师测试覆盖跨分块的回答位置、候选顺序、完整前文缓存、全词表归一化、上下文预算和小数组通信。基础测试使用 NumPy 模拟；已安装 Torch 和 Transformers 时，还会在 CPU 上创建随机初始化的小型 Qwen3，比较分块评分与整段前向结果，不下载模型：

```bash
python -B -m unittest train.test_teacher_client -v
```

KL 测试在已安装训练依赖的环境使用 CPU 张量，检查剩余词概率桶的 KL 数值以及反向梯度：

```bash
python -B -m unittest train.test_loss_regularization -v
```

无需模型和 GPU 的加载路由测试，覆盖原始模型、最终 adapter、中间 checkpoint、tokenizer 回退和 vLLM rank 容量：

```bash
python -B -m unittest discover -s eval -p test_checkpoint_loading.py -v
```

在安装评分依赖的 Linux 环境检查数学判定与本地数据集参考答案：

```bash
python -B -m unittest discover -s eval -p test_answer_grader.py -v
```

加载路由测试使用模拟后端，不能替代实际 GPU 推理。训练内评估链路仍未适配，保持 `eval_strategy="no"`，通过独立评测入口比较 checkpoint。

## GRPO LoRA 训练（双卡 A100 80G）

[train/grpo_train.py](train/grpo_train.py) 是独立的 GRPO 入口，默认加载本地
`/models/qwen3-1.7b`，读取 `dataset/train/dapo-17k.parquet`（17,398 题）。
使用 `prompt` 构造问题，`reward_model.ground_truth` 仅用于答案奖励；不把标准答案或
`solution` 放入模型提示。奖励为数学等价判定的 0/1 分，复用项目评测器，只提取
思考段落之后的最终 `\boxed{...}`，没有最终答案时得 0 分。

在现有 Linux/CUDA 训练环境中，从项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m train.grpo_train
```

该入口由两张卡分别运行训练副本及 colocate vLLM（每个引擎 TP=1），无需启动
`vllm/start_vllm.sh`。沿用 `requirements.txt` 及服务器现有的定制 vLLM 0.19.1，
不需要安装 FlashAttention。vLLM 每卡显存预算默认为 35%；实际峰值需要在服务器上验证。

| 参数 | 默认值 |
| --- | --- |
| 训练步数 | 1,000 个优化器更新步 |
| LoRA | r=32、alpha=64、all-linear、dropout=0 |
| 每卡训练 microbatch / 梯度累积 | 2 / 32 |
| 有效 batch / 每题回答数 | 128 条回答 / 8 |
| 每个更新步采样的题目数 | 16 |
| 学习率 / KL beta / clip epsilon | 5e-6 / 0.04 / 0.2 |
| 学习率调度 | 前 20 步 warmup，之后恒定 |
| prompt / completion 上限 | 1,024 / 4,096 tokens |
| 精度及省显存 | BF16、SDPA、gradient checkpointing |
| 保存频率 | 每 50 个优化器更新步，保留所有 checkpoint |

经典参数参考 [DeepSeekMath §4.2](https://arxiv.org/html/2402.03300v3#S4.SS2)：
原文采用 lr=1e-6、KL=0.04、每题 64 条回答、batch=1024、每轮采样更新一次。
这里将采样规模缩小到双卡预算，LoRA 学习率取 5e-6 作为起点。
论文没有统一规定训练步数；本脚本的 1,000 步对应约 16,000 次题目采样，
约为原始数据量的 0.92 倍，是本项目的预算配置，仍应通过独立评测选择 checkpoint。

显式指定 `loss_type="grpo"` 和组内奖励标准化；可用 `--loss-type dapo` 选择
[TRL 1.13 的 token 归一化损失](https://huggingface.co/docs/trl/v1.13.0/en/grpo_trainer)，
使用 DAPO 数据不等于复现完整 DAPO 算法。超长题目按实际 chat template token 数过滤，
不截断题目；到达生成长度上限的回答不参与 loss。默认关闭 Qwen3 的 `<think>` 模式，
仍要求逐步解题；如需开启，可加 `--enable-thinking --max-completion-length 8192`，
并重新验证显存和回答截断比例。其他参数见 `python -m train.grpo_train --help`。

保存路径为 `train/grpo_checkpoint/checkpoint-50/`、`checkpoint-100/` 等；
其中保存 LoRA、tokenizer 及恢复训练所需的 optimizer/scheduler/RNG 状态，不复制完整基座。
训练结束额外保存 `train/grpo_checkpoint/final/`，运行参数记录在 `grpo_run_config.json`。
断点续训命令如下，或把 `latest` 换为具体的 `checkpoint-<step>` 路径：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m train.grpo_train --resume-from-checkpoint latest
```

CPU 数据/奖励回归检查：`python -B -m unittest train.test_grpo_train -v`。
未安装 Math-Verify 时跳过实际数学等价检查；其余测试不依赖模型和 GPU。
