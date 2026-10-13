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
  student_generation.py      # 共卡 vLLM 生成、实时权重同步、回答 padding
  test_student_generation.py # 权重同步顺序与回答对齐测试
  utils.py                   # 消息准备、tokenization、padding
  teacher_client.py          # Transformers 教师子进程与 GPU 候选评分
  test_teacher_client.py     # 分块评分对齐与通信测试
  outputs/<model_name>/      # 最终 LoRA 与 tokenizer
  checkpoint/<model_name>/   # checkpoint-<global_step>/
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

学生与教师模型路径的默认值分别在 [train/myconfig.py](train/myconfig.py) 的 `student_model_path`、`teacher_model_path` 中设置。命令行 `--student_model_path`、`--teacher_model_path` 可分别覆盖默认值，也支持连字符写法 `--student-model-path`、`--teacher-model-path`；修改学生路径不会改变教师路径。两个 system prompt、`LoraConfig(...)` 和其他训练参数在 [train/main.py](train/main.py) 中修改。`teacher_gpu="1"` 指定教师使用的物理卡号；其他教师配置在 `myconfig.py` 中设置，也可在 `MyConfig(...)` 中覆盖。

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
| `student_vllm_gpu_memory_utilization` | 0.5 | 生成时 vLLM 的显存预算占 GPU0 总显存的比例 |
| `learning_rate` | 1e-5 | 学习率 |
| `num_train_epochs` | 1 | 遍历数据的轮数 |
| `max_new_tokens` | 2048 | 每次学生回答的生成上限 |
| `loss_top_k` | 16 | 每个有效回答位置参与 loss 的候选数 |
| `save_steps` | 50 | 每50个优化器更新步保存 checkpoint |
| `logging_steps` | 10 | 日志间隔 |
| `eval_strategy` | `no` | 训练内评估关闭，使用独立评测入口 |
| LoRA | r=16，alpha=32，all-linear | 当前 adapter 配置 |

### 启动 teacher 与学生

直接启动训练。GPU0 上的学生使用共卡 vLLM 生成；训练脚本自动创建 GPU1 教师子进程，通过 Transformers 加载冻结模型，无需另开服务：

```bash
wandb login
CUDA_VISIBLE_DEVICES=0 python -m train.main
```

通过命令行分别指定学生和教师（不传的路径使用 `myconfig.py` 中的默认值）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.main \
  --student_model_path /models/student \
  --teacher_model_path /models/teacher
```

当前 `report_to="wandb"`，在线记录需要登录；需要离线记录时可设置 `WANDB_MODE=offline`。项目名可通过 `WANDB_PROJECT` 设置，运行名在 `main.py` 的 `run_name` 中设置。

学生进程只暴露 GPU0，教师子进程独立设置 `CUDA_VISIBLE_DEVICES=1`，使用 `spawn` 启动，避免与学生的 CUDA 上下文及 Trainer 多卡包装冲突。改变学生卡号时修改启动命令；改变教师卡号时修改 `teacher_gpu`，两者都是机器或容器可见的卡号，不是父进程重映射后的编号。当前训练面向单个学生进程，不使用 `torchrun`。

学生生成使用 TRL 1.13.0 的 `VLLMGeneration`，`mode="colocate"`、`tensor_parallel_size=1`。最大并发数跟随 `per_device_train_batch_size`，当前是32；上下文上限为 `max_length + max_new_tokens = 3072`。vLLM 直接接收去除 padding 后的 prompt IDs，每题生成一个回答，仅返回 token IDs，不请求 logprob 或解码文本。[TRL 实现](https://github.com/huggingface/trl/blob/v1.13.0/trl/generation/vllm_generation.py)

每次生成前都调用 `sync_weights()`：把当前学生的 LoRA 临时合并后复制进 vLLM，再恢复学生的未合并训练状态，并清空旧前缀缓存。首次生成、每次优化器更新之后的下一次生成以及恢复训练后的首次生成都经过这一同步。不会从磁盘重新加载旧 adapter。生成后引擎进入 level 2 睡眠，释放权重和 KV cache 显存；下次生成前重新同步。因此参数更新和生成之间没有使用旧权重的窗口，也不需要在 `optimizer.step()` 后立即唤醒空闲引擎。

`student_vllm_gpu_memory_utilization=0.5` 在 `myconfig.py` 中调整。它是生成阶段的显存预算，不是 GPU 计算利用率目标；A100 80G 上约为一半总显存，其余留给同卡学生模型等开销。生成阶段与训练前向阶段交替执行。当前按配置让32条一起计算 loss，没有分组或隐藏的梯度累积；长回答下的全词表 logits 和计算图仍可能超过80G，vLLM 睡眠不能消除学生前向本身的显存需求。

教师当前 `teacher_max_model_len=8192`、`teacher_score_chunk_size=8192`，在上下文预算内相当于整段评分。按学生最多生成2048个 token 计算，教师提示（含参考解答）可使用最多6144个 token；教师只评分，不额外生成 token。超过总长度预算时会报错，不自动截断。教师模型也需要支持该长度。

教师仍逐条处理 batch 中的输入。`teacher_score_chunk_size` 可调小以限制单次前向的 logits 显存，用 KV cache 保留前文；它不改变上下文长度、候选数或 loss。教师没有固定显存比例或预分配 KV cache 配额，显存由模型权重、实际长度的 KV cache 和当前分块 logits 决定。

### 训练目标与资源开销

每个 batch 先由学生生成回答，再进行带梯度的学生前向，得到每个有效回答位置的 top-K 候选 ID。随后同一个 teacher 对普通上下文和特权上下文分别打分。只在有效回答位置计算：

```text
indices = topk(student_logits, K).indices
r_1 = teacher_logprob[indices] - normal_teacher_logprob[indices]
r2 = softmax(student_topk_logits.detach()) * r_1
loss = mean_over_answer_tokens(-sum_over_K(r2 * student_topk_logprob))
```

`student_topk_logprob` 来自全词表 `log_softmax`，保留梯度；teacher 差值和权重不保留梯度。教师在 GPU1 上用 `selected_logits - logsumexp(full_vocab_logits)` 得到候选 logprob，全词表归一化保持不变。只把 `[有效回答 token 数, K]` 的 float32 数组通过本地 Pipe 返回，训练进程相减后将 `r_1` 搬到 GPU0。这是实验性目标，loss 可以为负。

学生 vLLM 生成不保留梯度，生成后根据返回长度补齐回答、构造 mask 和 labels；真实 EOS 参与 loss，回答 padding 和 prompt 不参与。计算 loss 的 Transformers 学生前向显式关闭 KV cache，继续由 Trainer 执行反向传播与优化器更新。teacher 使用固定的基座权重，不随学生更新参数。教师评分期间，学生前向的计算图仍保留在 GPU0 上。

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
CUDA_VISIBLE_DEVICES=0,1 python eval/run_eval.py \
  --model /models/qwen --tensor-parallel-size 2
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

学生生成测试覆盖首次/更新后/恢复后的同步顺序、同步失败时禁止生成、批量配置和 prompt/回答/EOS/padding 对齐。安装 Torch、TRL、PEFT 时还会用 CPU 小模型检查 TRL 的 LoRA 合并同步和恢复；不启动 vLLM，也不下载模型：

```bash
python -B -m unittest train.test_student_generation -v
```

教师测试覆盖跨分块的回答位置、候选顺序、完整前文缓存、全词表归一化、上下文预算和小数组通信。基础测试使用 NumPy 模拟；已安装 Torch 和 Transformers 时，还会在 CPU 上创建随机初始化的小型 Qwen3，比较分块评分与整段前向结果，不下载模型：

```bash
python -B -m unittest train.test_teacher_client -v
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
