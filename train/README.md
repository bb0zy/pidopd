# 使用 `solution` 作为特权信息的训练

`train_direct_opd.py` 使用同一个冻结参考模型进行两次评分：一次只看题目，一次看题目和参考解答。学生生成与训练时只看题目。这个版本沿用原脚本的 top-1 学生候选、LoRA、采样 KL 约束和自适应 `alpha`；教师差值的含义已经改变，不是论文中 RL 后教师与 RL 前参考模型的差值，不能将结果视作原论文算法的复现。

## 数据与打分

默认依次读取 `dataset/train/train-00000-of-00005.parquet` 和 `dataset/train/train-00001-of-00005.parquet` 的 `problem`、`solution` 字段，保持各文件内的行顺序。不要使用数据里的 `messages` 字段作为学生输入，因为其中包含 assistant 解答。可用 `--data FILE1.parquet FILE2.parquet` 指定其他有相同字段的文件，文件顺序就是训练顺序；`--prompt-key` 和 `--solution-key` 可修改字段名。默认 `--prompt-format existing` 将题目原样作为单条 user 消息；`--prompt-format dapo` 则应用原有的 `Answer:` 模板。

教师专用提示词将 `solution` 附加到题目末尾，并明确标记为参考解答。教师和无特权参考使用同一份 `--teacher-ref` 权重，但输入上下文不同。对学生采样回答的每个位置，选学生分布的 top-1 token `v*`，计算

```text
r_t = log p_ref(v* | problem, solution, student_response_<t)
    - log p_ref(v* | problem, student_response_<t)
```

两项均在相同的学生回答 token 上对齐。`solution` 不进入学生 rollout、学生梯度前向或学生初始 anchor。此目标可能让学生学习参考解答的风格与步骤；`solution` 的质量会直接影响训练信号。

超出 `--max-prompt-tokens`（默认 1024）的普通提示词或 `--max-teacher-prompt-tokens`（默认 8192）的教师提示词会被跳过，数量写入运行目录 `config.json`。两侧各自的提示词上限加 `--max-new-tokens`（默认 2048）必须在相应模型上下文长度内。`solution` 不会被静默截断。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0,1 python train/train_direct_opd.py \
  --student STUDENT_DIR \
  --teacher-ref REF_DIR \
  --student-device cuda:0 \
  --teacher-ref-device cuda:1
```

模型目录名相对于 `/models`，也可使用绝对路径或通过 `--models-root` 修改根目录。模型和数据只从本地读取。学生与参考模型的 token-to-ID 映射及输出词表大小必须相同。单 GPU 时可将两者都放在 `cuda:0`，但需确认显存足够。

一个 optimizer step 默认按文件顺序取 64 道题，每题生成 1 条回答，共 64 条 rollout。同一次训练中，各步取不同数据行，不循环使用题目。两个默认分片各有 17,824 行，合计 35,648 行，题目文本无重复；覆盖默认 300 步需至少 19,200 条通过长度过滤的样本。`--steps` 默认 300，是步数上限；若过滤后数据不足，最后一批可少于 64 题，数据用完即结束并保存最后一个检查点。

生成、教师评分和学生反向传播均按长度分组后批量执行。默认 `--rollout-batch-size 8`、`--score-batch-size 4`、`--train-microbatch-size 4`。生成采用左侧 padding，评分和训练采用右侧 padding 与 attention mask；loss 只取真实回答 token，按整步全部回答 token 加权。所有训练 microbatch 累积梯度后只做一次 optimizer step。每步在 `metrics.jsonl` 记录 `rollout_seconds`、`score_seconds` 和 `train_seconds`，便于找出瓶颈。批量大小越大通常吞吐越高，但长 `solution` 的教师评分会显著增加显存占用；显存不足时先降低 `--score-batch-size`，再降低 `--train-microbatch-size` 和 `--rollout-batch-size`。参考模型只加载一次，但条件与无条件打分仍各需一次前向；这里未接入 vLLM。训练是单进程 LoRA，不能用 `torchrun` 启动。

检查点在 `train/outputs/<UTC时间>/checkpoint-N`，默认每 20 步及最后一步保存。恢复时传入原始学生与参考模型，以及 `--resume` 和更大的 `--steps`；必须保持数据文件及行顺序不变，才能从正确位置接着取题。三个批量大小可以在恢复时调整；其他核心参数需与原运行一致。旧版随机抽题或使用 RL 前后两份教师模型的检查点不能用此版本恢复。

```bash
python train/train_direct_opd.py \
  --student STUDENT_DIR --teacher-ref REF_DIR \
  --resume train/outputs/RUN/checkpoint-100 --steps 200

python train/merge_lora.py \
  --checkpoint train/outputs/RUN/checkpoint-200 \
  --output train/outputs/RUN/merged-200

python eval/run_eval.py --model train/outputs/RUN/merged-200
```
