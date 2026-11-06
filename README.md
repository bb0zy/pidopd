# pidopd

数学模型训练与离线评测，包含分布蒸馏、Direct-OPD LoRA 训练及 GRPO / DAPO 全量微调。
所有命令从项目根目录执行，面向 Linux / NVIDIA GPU 服务器。

## 环境

Python 3.12、CUDA 12.8，依赖版本见 [requirements.txt](requirements.txt)。

```bash
python -m pip install -r requirements.txt
python -m pip check
```

vLLM 使用服务器现有的定制版 0.19.1，需兼容 Transformers 5.3.0；
未包含在安装清单中，单独安装。模型与数据均从本地读取。

下方 GRPO、DAPO 和 Align 命令假设虚拟环境位于 `.venv`，请按实际路径替换
`.venv/bin/python`，无需先激活环境。多卡通过该 Python 执行
`-m torch.distributed.run`（等价于 `torchrun`），确保使用指定虚拟环境。

GRPO 与 DAPO 的公共训练流程和奖励函数集中在 [train/rl_train.py](train/rl_train.py)，
各入口保留各自参数和算法逻辑；两阶段 KD 复用其中的设备选择与完整模型续训检查。

## 教师分布对齐（两阶段 KD）

教师 `/models/qwen3-4b-1`，学生 `/models/qwen3-1.7b`。
先用 [teacher_rollout.py](train/teacher_rollout.py) 对 `dataset/train/dapo-17k.parquet`
批量生成教师回答，再用 [align_train.py](train/align_train.py) 纯对齐，不再 rollout。

```bash
# 第一阶段：vLLM，每题默认1条；改为 --num-rollouts 4 可生成4条
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.teacher_rollout

# 第一阶段也支持双卡数据并行，每卡各加载一份教师并分摊题目
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m train.teacher_rollout --data-parallel-size 2

# 第二阶段：单卡全量分布对齐
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.align_train

# 第二阶段也支持双卡 DDP
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module train.align_train
```

第一阶段保存到 `dataset/train/dapo-17k-teacher.jsonl`，包含原始 token ID 和结束原因；
同名 `.meta.json` 记录配置及截断率。默认回答预算16,384 tokens、上下文32,768、
prompt 上限1024，超长题目过滤。生成温度0.7、top-p=0.8、top-k=20，默认关闭 thinking。
增加预算示例：`--max-new-tokens 32768 --max-model-len 40960`，需模型支持；
第二阶段相应设置 `--max-length 40960`。达到长度上限的回答会保存，但对齐时过滤，完整回答不裁剪。
生成续跑追加 `--resume`，只补生成缺失回答，支持续跑时更换卡数。
第一阶段使用普通 Python 启动；多卡设置 `--data-parallel-size <卡数>` 及对应可见 GPU。
各进程先写 `.parts/` 分片，结束后自动合并到同一 JSONL，Align 命令不变。
`--batch-size` 和 `--max-num-seqs` 按每个实例计算；TP 默认1，原有 `--tensor-parallel-size` 仍可使用。

第二阶段采用 **学生 top-k 蒸馏**（默认 `--loss-top-k 16`），复用 Direct-OPD 的候选选择及加权函数。
在缓存教师回答的每个位置，由学生选择 top-k 候选；教师对相同候选评分。
两边 log 概率都以全词表归一化，权重在学生 top-k 内重新归一化并停止梯度。
单教师蒸馏信号是 `log p_teacher − log p_student`，对应反向 KL 的 top-k 梯度近似；
它替代了原来的全词表前向 KL，而非硬标签 SFT。覆盖整条回复及 EOS，prompt 和 padding 不参与损失。
这里沿用教师缓存轨迹、不做学生 rollout、不使用双教师差分，也没有额外的初始学生 KL 约束。
返回的 loss 是策略梯度代理损失，可以为负，不能作为非负 KL 指标解释。

学生仍是全量微调：FP32 参数配合 BF16 混合精度，教师 BF16 冻结。
默认学习率 `1e-5`、1 epoch、每卡 batch=1、梯度累积32、3% warmup、最大长度32,768。
输出层按 `--logit-chunk-size 128` 个回复位置分块计算，并在反向传播时重算，避免保存整段全词表 logits；
top-k 只限制蒸馏候选，完整词表的归一化和输出层计算仍然需要。可降低 chunk size 减少 logits 峰值显存。
设置 `--logit-chunk-size -1` 时，当前 microbatch 的所有有效回复位置一次计算，关闭输出层分块和重算；
prompt 与 padding 仍被屏蔽，默认值仍为128。此模式会同时保存全部有效回复位置的全词表 logits，显存占用更高。
这条训练路径不再依赖 Liger kernel，旧的 `--use-liger-kernel` / `--no-use-liger-kernel` 参数已移除。
单卡默认约544次更新，数量随回答条数及过滤变化。示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.align_train --loss-top-k 16 --logit-chunk-size 128
```

每100步保存完整学生及续训状态到 `train/align_checkpoint/checkpoint-<step>/`，最终模型在 `final/`；
续训追加 `--resume-from-checkpoint latest`。自定义生成文件需同时指定第一阶段 `--output-path`
和第二阶段 `--data-path`。其他参数见各入口 `--help`。
评测可用 `python eval/run_eval.py --checkpoint-dir train/align_checkpoint`。

## GRPO 全量微调

入口：[train/grpo_train.py](train/grpo_train.py)。
默认模型为 `/models/qwen3-1.7b`，数据为 `dataset/train/dapo-17k.parquet`（17,398 题）。
使用最终 `\boxed{...}` 答案的数学正确性作为 0/1 奖励，标准答案不进入模型提示。

```bash
# 单卡
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m train.grpo_train

# 双卡 DDP；更多卡时调整可见卡列表和进程数
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module train.grpo_train
```

每张卡运行完整训练模型及 colocate vLLM（TP=1），无需单独启动 vLLM 服务。
默认开启 vLLM sleep mode，vLLM 引擎显存预算为每卡 50%；目标资源为 A100 80G。
这不是整个训练进程的显存上限；实际峰值随回答长度变化，需在服务器实测。

| 参数 | 默认值 |
| --- | --- |
| 更新步数 / 每题回答数 | 1,000 / 8 |
| 每卡 microbatch / 梯度累积 | 4 / 16 |
| 有效 batch | 64 × GPU 数条回答 |
| 学习率 / KL 系数 | 1e-6 / 0.04 |
| prompt / completion 上限 | 1,024 / 4,096 tokens |
| 精度 / thinking 模式 | BF16 / 关闭 |
| checkpoint 保存 | 每 100 步，保留全部 |

默认从 `2 × 32` 调为 `4 × 16`，保持每卡每步64条回答，提高单次训练的并行量。
以接近70G峰值为调参目标；若仍有余量可试 `--per-device-train-batch-size 8 --gradient-accumulation-steps 8`。
若训练 OOM，退回 `2 / 32`；若生成或唤醒 vLLM 时 OOM，降低 `--vllm-gpu-memory-utilization`。

超长题目过滤、不截断；生成达到长度上限的回答不参与 loss。
其他选项见 `python -m train.grpo_train --help`。
参数参考 [DeepSeekMath §4.2](https://arxiv.org/html/2402.03300v3#S4.SS2)，
训练步数和采样规模按本项目设置。

完整模型、tokenizer 和续训状态保存在 `train/grpo_checkpoint/checkpoint-<step>/`；
最终模型在 `train/grpo_checkpoint/final/`。单卡、多卡续训均在原命令后追加：

```bash
--resume-from-checkpoint latest
```

也可将 `latest` 换为具体的 `train/grpo_checkpoint/checkpoint-<step>` 路径。
可用 `--output-dir` 分开保存不同实验。单卡若要保持双卡默认的有效 batch，
设置 `--gradient-accumulation-steps 32`。

## DAPO 全量微调

入口：[train/dapo_train.py](train/dapo_train.py)，模型、数据和 system prompt 与 GRPO 相同。
默认非思考模式、最多生成 **10,240 tokens**，训练1,000步；每100步保存完整模型及续训状态到
`train/dapo_checkpoint/checkpoint-<step>/`，最终模型在 `final/`。

```bash
# 单卡
CUDA_VISIBLE_DEVICES=0 python -m train.dapo_train

# 双卡 DDP
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module train.dapo_train
```

包含 DAPO token loss、上下裁剪0.2/0.28、动态采样（过滤全对/全错题组并补足）和超长惩罚。
默认 KL=0、学习率1e-6、每题8条；针对长回答使用每卡 microbatch=1、累积64，
每步仍为每卡64条筛选后的回答。vLLM colocate TP=1、sleep mode 和50%引擎显存预算与 GRPO 一致。
超过8192 tokens 后线性扣分，在10240处达到-0.5（对应本项目0/1正确性奖励尺度）；截断回答不参与 loss。
动态采样最多16轮，未补足时明确报错，可用 `--max-sampling-rounds` 调整。

续训追加 `--resume-from-checkpoint latest`；长度用 `--max-new-tokens` 或 `--max-completion-length`；
保留 KL 可设置 `--beta 0.04`。日志 `dapo/raw_accuracy` 是筛选前正确率，
`dapo/group_acceptance_rate` 是题组保留率；`reward` 则是筛选后含长度惩罚的奖励，不能直接当作正确率。
动态采样依赖项目固定的 `trl==1.13.0`，多卡同步和显存峰值仍需在服务器验证。
评测使用 `python eval/run_eval.py --checkpoint-dir train/dapo_checkpoint --checkpoint 100`。

## Direct-OPD：双教师 LoRA 训练

入口：[train/main.py](train/main.py)，损失见 [train/direct_opd.py](train/direct_opd.py)。
使用与 beautifulhand/pidopd 相同的教师信号：`log π_after − log π_before`。
两个冻结教师读取完全相同的学生 prompt 和 rollout，不再使用参考解答提示。
默认读取 `dataset/train/dapo-17k.parquet` 的 `prompt` 列，也支持旧数据的 `problem` 列；不读取答案。
prompt 使用与 GRPO/DAPO 相同的最终答案指令，要求末行仅包含 `\boxed{...}`，默认关闭 thinking；超长 prompt 过滤、不截断。

GPU0 训练学生，GPU1 运行 vLLM 学生服务、两个教师及冻结的初始学生（KL 锚点）。
显存不足时可将教师放在第三张卡，例如训练命令增加 `--teacher-gpu 2`，vLLM 仍在 GPU1。
该入口使用单个训练进程，使用普通 Python 启动。

先在一个终端启动 vLLM 服务，`MODEL_PATH` 必须是训练学生对应的完整基础模型或完整模型 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=1 MODEL_PATH=/models/qwen3-1.7b bash vllm/start_vllm.sh
```

在另一终端启动训练，教师路径可替换为本地完整模型、完整 checkpoint 或 LoRA adapter 目录：

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.main \
  --student-model-path /models/qwen3-1.7b \
  --model-before /models/DeepSeek-R1-Distill-Qwen-1.5B \
  --model-after /models/JustRL-DeepSeek-1.5B \
  --output-dir train/checkpoint/direct-opd
```

学生 rollout 始终使用 vLLM。每次生成前同步当前学生权重（合并 LoRA 后传输，再恢复训练模型），
清空 prefix cache；首次生成与 checkpoint 续训后也执行同步。默认每题生成4条、温度1、top-p=1，
采样不限制 top-k；`--loss-top-k 16` 只控制损失候选集合。
两个教师对学生候选给出全词表归一化的 log 概率，按 token 字符串映射不同模型的 token ID；
遇到教师词表无法表达的学生 token 明确报错，避免把不同 token 的概率直接相减。

LoRA 默认为 r=16、alpha=32、dropout=0，作用于 q/k/v/o 与 gate/up/down。
损失为 top-k 概率加权的教师 log-ratio 加采样 token 上的 k3 KL，KL 始终锚定本次实验的初始学生。
默认 α=1，ε=0.01，范围 [0.5, 2.5]。自适应系数按一次 optimizer step 内所有 microbatch
的 token 加权信号更新，供下一步使用；对方实现先计算当前整批信号，再更新 α 用于当前步，时序有一拍差别。
固定系数用 `--no-adaptive-kl --kl-beta <值>`；完全关闭 KL 用 `--no-adaptive-kl --kl-beta 0`。

默认学习率1e-4、180步、prompt/回答上限1024/1024，每45步保存一次。
训练 microbatch 参数以 prompt 数计：默认1条 prompt × 每题4次生成 × 累积16次 = 每步64条回答；
多条 rollout 会一起进行训练前向，需要按显存调整 `--num-generations` 与 batch。
对方记录的实验每步60条 prompt、每题4次生成；本入口可用 `--gradient-accumulation-steps 60` 得到相同的回答条数。
更多选项见 `python -m train.main --help`；默认不连接 W&B，按需加 `--report-to wandb`。

checkpoint 位于 `train/checkpoint/direct-opd/checkpoint-<step>/`，包含 adapter、tokenizer、
Trainer/优化器/调度器/RNG 状态，以及 `direct_opd_state.json` 中的 α、vLLM 请求计数和实验配置。
最终 adapter 在 `train/checkpoint/direct-opd/final/`。在原训练命令后追加：

```bash
--resume-from-checkpoint latest
# 或 --resume-from-checkpoint train/checkpoint/direct-opd/checkpoint-45
```

续训保留原来的 `--student-model-path` 和教师路径，恢复 LoRA、优化器、调度器、RNG 和 α；
vLLM 的采样 seed 由保存的请求计数恢复。配置不一致或 checkpoint 缺少续训状态时明确报错。
旧版 prompt 差分训练 checkpoint 或对方手写循环的 checkpoint 不能直接恢复 Trainer 状态；
可用 `--student-model-path <adapter目录>` 将其作为新实验的初始学生，重建优化器和 KL 锚点。
adapter 的 `adapter_config.json` 必须指向可访问的本地完整基础模型；服务用该基础模型启动。
学生的完整模型 checkpoint 同样可以直接作为 `--student-model-path`。
评测继续使用 `eval/run_eval.py --model <基础模型> --lora-path <adapter目录>`。

## 评测

入口：[eval/run_eval.py](eval/run_eval.py)。默认使用 vLLM，评测 `dataset/eval/` 下的
AIME24、AIME25 和 HMMT；每题采样 32 次，使用 Math-Verify 判定答案等价。
默认按步数顺序评测 `train/grpo_checkpoint/checkpoint-*` 中所有完整 checkpoint，
每个模型使用独立进程，结果分别保存，并生成总汇总。批量模式默认关闭 thinking，
可用 `--chat-template-kwargs` 覆盖。

```bash
# 评测全部 GRPO checkpoint
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py

# 仅评测最新 checkpoint；指定步数时改为 --checkpoint 100
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --checkpoint latest

# 单独评测原始模型或 final 完整模型
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --model train/grpo_checkpoint/final
```

常用选项：`--checkpoint-dir <检查点父目录>`、`--backend transformers`、`--data <文件或目录>`、`--limit 5`、
`--samples 16`、`--chat-template-kwargs '{"enable_thinking":false}'`。
评测 GRPO 模型时应与训练的 thinking 设置一致；不同实验保持相同的采样和长度预算。

结果写入 `eval/results/<UTC时间>/`，批量模式按 checkpoint 分目录，顶层 `summary.json`
汇总各模型指标。显式指定 `--model` 时仍可搭配 `--lora-path` 评测教师训练的 adapter。
`avg@32` 表示平均采样正确率；使用 16 次采样时字段名称仍为 `avg@32`，
实际分母为 16。完整参数见 `python eval/run_eval.py --help`。

## 检查

单测统一放在 `tests/`：训练相关测试在 `tests/train/`，评测相关测试在 `tests/eval/`。
从项目根目录执行全部测试：

```bash
python -B -m unittest discover -s tests -t . -v
```

也可以按模块执行，例如：

```bash
python -B -m unittest tests.train.test_align_train -v
python -B -m unittest tests.eval.test_answer_grader -v
```

测试依赖对应的训练或评分库，不能替代实际 GPU 训练与推理验证。
