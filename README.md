# pidopd

数学模型在线训练与离线评测，包含教师引导的 LoRA 训练和独立的 GRPO 全量微调。
所有命令从项目根目录执行，面向 Linux / NVIDIA GPU 服务器。

## 环境

Python 3.12、CUDA 12.8，依赖版本见 [requirements.txt](requirements.txt)。

```bash
python -m pip install -r requirements.txt
python -m pip check
```

vLLM 使用服务器现有的定制版 0.19.1，需兼容 Transformers 5.3.0；
未包含在安装清单中，单独安装。模型与数据均从本地读取。

## GRPO 全量微调

入口：[train/grpo_train.py](train/grpo_train.py)。
默认模型为 `/models/qwen3-1.7b`，数据为 `dataset/train/dapo-17k.parquet`（17,398 题）。
使用最终 `\boxed{...}` 答案的数学正确性作为 0/1 奖励，标准答案不进入模型提示。

```bash
# 单卡
CUDA_VISIBLE_DEVICES=0 python -m train.grpo_train

# 双卡 DDP；更多卡时调整可见卡列表和进程数
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --module train.grpo_train
```

每张卡运行完整训练模型及 colocate vLLM（TP=1），无需单独启动 vLLM 服务。
默认开启 vLLM sleep mode，显存预算为每卡 25%；目标资源为 A100 80G。

| 参数 | 默认值 |
| --- | --- |
| 更新步数 / 每题回答数 | 1,000 / 8 |
| 每卡 microbatch / 梯度累积 | 2 / 32 |
| 有效 batch | 64 × GPU 数条回答 |
| 学习率 / KL 系数 | 1e-6 / 0.04 |
| prompt / completion 上限 | 1,024 / 4,096 tokens |
| 精度 / thinking 模式 | BF16 / 关闭 |
| checkpoint 保存 | 每 100 步，保留全部 |

超长题目过滤、不截断；生成达到长度上限的回答不参与 loss。
其他选项见 `python -m train.grpo_train --help`。
参数参考 [DeepSeekMath §4.2](https://arxiv.org/html/2402.03300v3#S4.SS2)，
训练步数和采样规模按本项目设置。

完整模型、tokenizer 和续训状态保存在 `train/grpo_checkpoint/checkpoint-<step>/`；
最终模型在 `train/grpo_checkpoint/final/`。续训时在原命令后追加：

```bash
--resume-from-checkpoint latest
```

可用 `--output-dir` 分开保存不同实验。单卡若要保持双卡默认的有效 batch，
设置 `--gradient-accumulation-steps 64`。

## 教师引导的 LoRA 训练

入口：[train/main.py](train/main.py)，配置见 [train/myconfig.py](train/myconfig.py)。
学生默认 `/models/qwen3-1.7b`，教师默认 `/models/qwen3-0.6b`；
读取 `dataset/train/train-00000-of-00002.parquet` 和 `train-00001-of-00002.parquet`。

GPU0 训练学生，GPU1 运行学生 vLLM 服务和冻结教师。先在一个终端启动服务：

```bash
CUDA_VISIBLE_DEVICES=1 MODEL_PATH=/models/qwen3-1.7b bash vllm/start_vllm.sh
```

服务就绪后，在另一终端训练（W&B 需提前登录，或设置 `WANDB_MODE=offline`）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m train.main \
  --student_model_path /models/qwen3-1.7b \
  --teacher_model_path /models/qwen3-0.6b
```

学生与服务的模型路径必须一致；该入口使用单个训练进程。
教师比较普通提示和包含参考解答的提示，对学生 top-K 候选评分，并添加相对初始学生的 KL 约束；
可用 `--kl_beta 0` 关闭 KL。

LoRA 默认为 r=16、alpha=32、all-linear，每 50 步保存 checkpoint：
`train/checkpoint/<模型名>/checkpoint-<step>/`。
最终 adapter 在 `train/outputs/<模型名>/`。

## 评测

入口：[eval/run_eval.py](eval/run_eval.py)。默认使用 vLLM，评测 `dataset/eval/` 下的
AIME24、AIME25 和 HMMT；每题采样 32 次，使用 Math-Verify 判定答案等价。

```bash
# 原始模型
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --model /models/qwen3-1.7b

# GRPO 完整模型或 checkpoint
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py --model train/grpo_checkpoint/final

# LoRA adapter 或 checkpoint
CUDA_VISIBLE_DEVICES=0 python eval/run_eval.py \
  --model /models/qwen3-1.7b --lora-path train/outputs/qwen3-1.7b
```

常用选项：`--backend transformers`、`--data <文件或目录>`、`--limit 5`、
`--samples 16`、`--chat-template-kwargs '{"enable_thinking":false}'`。
评测 GRPO 模型时应与训练的 thinking 设置一致；不同实验保持相同的采样和长度预算。

结果写入 `eval/results/<UTC时间>/`，包含配置、逐题生成记录及汇总指标。
`avg@32` 表示平均采样正确率；使用 16 次采样时字段名称仍为 `avg@32`，
实际分母为 16。完整参数见 `python eval/run_eval.py --help`。

## 检查

```bash
python -B -m unittest train.test_grpo_train -v
python -B -m unittest train.test_student_generation -v
python -B -m unittest train.test_teacher_client -v
python -B -m unittest train.test_loss_regularization -v
python -B -m unittest discover -s eval -p test_checkpoint_loading.py -v
python -B -m unittest discover -s eval -p test_answer_grader.py -v
```

测试依赖对应的训练或评分库，不能替代实际 GPU 训练与推理验证。
