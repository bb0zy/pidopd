# Direct-OPD：按论文实现的 top-1 简化训练

依据 `related_work/dopd.pdf`：*Weak-to-Strong Generalization via Direct On-Policy Distillation*，重点为 §2.3、§2.4 和附录 A。用户要求将论文 top-16 简化成 top-1，并使用 TRL/Transformers。代码尚未运行、导入验证或进行训练测试，不能宣称复现了论文报告的实验分数。

## 模型与实现范围

运行时必须传入三个本地目录：

| 参数 | 含义 |
| --- | --- |
| `--student` | 待训练学生的初始模型 |
| `--teacher` | 已完成 RL 的教师 |
| `--teacher-ref` | 同一教师在该次 RL 之前的模型 |

目录名默认相对 `/models` 解析，也可以传绝对路径；可通过 `--models-root` 指定其他模型根目录。没有默认具体模型，也不会从 Hugging Face 下载模型或数据。

学生使用 PEFT LoRA 微调，基础权重冻结。学生与参考共享同一份 base model：生成和训练时启用 adapter，计算 anchor 概率时在 `torch.no_grad()` 下使用 `with student.disable_adapter():`。离开该上下文后恢复 adapter，不单独加载 anchor，也不在训练途中合并 LoRA。因此内存中只有学生（base + LoRA）、教师和教师参考三个模型实例，已移除 `--anchor-device`。

只训练 LoRA A/B 参数，bias、embedding、lm_head 等原始参数均冻结；不使用 modules_to_save 或会改变基础权重的特殊初始化。新建 LoRA 使用默认零增量初始化，保证初始学生等于 base；恢复时仍加载原始 base，再恢复 adapter。教师和教师参考必须是一对有实际 RL 改进关系的检查点；本代码不负责训练教师，也不使用标准答案或稀疏正确性奖励训练学生。

TRL 提供 `apply_chat_template`，Transformers 负责本地模型加载、学生生成和模型导出；Direct-OPD 的自定义损失和优化循环由 PyTorch 实现。没有使用 GRPOTrainer 的组内标准化，因为这会改变论文的稠密 token 奖励方法；也不使用 verl 或 vLLM 训练后端。

## top-1 的精确定义

在学生自身随机生成的每个前缀上，选择学生概率最大的候选 `v* = argmax_v p_student(v)`。这个候选不一定是实际采样生成的 token；rollout 依然是 temperature=1、top_p=1、top_k=0 的随机生成，不是贪心生成。

教师差值使用完整词表归一化的 log 概率：

```text
r_t = log p_teacher(v* | prefix) - log p_teacher_ref(v* | prefix)
L_shift = - mean_response_tokens(stop_gradient(r_t) * log p_student(v* | prefix))
```

这是论文 Eq. 11–15 的单元素候选集版本：集合内归一化后 `p_bar(v*)=1`，因此权重为 `r_t`。不能对一个 logit 做 log_softmax 再当作学生 log 概率，否则它恒为 0，梯度也为 0。`objective.py` 通过完整词表 logsumexp 计算选中 token 的概率，按位置分块转 FP32，避免一次创建完整 FP32 softmax。top-1 并不能省掉基础语言模型的完整词表输出层计算。

学生初始 anchor 在实际采样 token 上提供冻结 log 概率；采用 sampled k3 KL surrogate：

```text
d_t = log p_anchor(y_t | prefix) - log p_student(y_t | prefix)
L_KL = mean_response_tokens(exp(d_t) - d_t - 1)
L = L_shift + alpha * L_KL
```

这属于常用采样 KL 惩罚实现，不是完整词表 KL 的精确梯度。论文称使用 verl 的标准 KL penalty，但没有在 PDF 中完整指定 estimator/聚合方式；这里明确选用 k3 与全批次 response-token 平均，不冒充与作者实现逐项一致。不对极端差值静默裁剪；非有限 loss/gradient 会终止训练。

默认启用论文 Eq. 16 的自适应系数：先计算整批次有效 response token 的平均 `r_t`，在反向前更新 `alpha = clip(alpha * (1 + 0.01 * sign(mean_r)), 0.5, 2.5)`。`--fixed-alpha` 可关闭，`--alpha` 控制初值/固定值，默认 1.0。

## 本地数据与默认参数

默认读取 `dataset/train/dapo-17k.parquet` 中的 `prompt` 字段，支持文本或以 user 结尾的 role/content 列表；其他字段名通过 `--prompt-key` 指定。默认按附录 A 加入 DAPO 的 `Answer:` 提示；多轮或已经格式化的提示可用 `--prompt-format existing` 保持原文。代码只删除已知 boxed 指令的精确后缀，不尝试猜测、清理任意模板。

超过 1024 prompt tokens 的题目会过滤，过滤数量写入 config；不会截断题目或把标准答案拼接到提示。每个 step 从保留题目中无放回随机选取 64 题，不同 step 可重复。这是随机训练采样，不是 eval 的全量遍历。

| 设置 | 默认值 |
| --- | --- |
| prompts / optimizer step | 64 |
| rollouts / prompt | 4，共 256 条回答 |
| prompt / response 上限 | 1024 / 2048 tokens |
| rollout temperature / top_p | 1.0 / 1.0 |
| 学习率 / 更新次数 | 1e-6 / 300 |
| 候选支持集 | top-1（用户指定，论文为 16） |
| 初始 alpha | 1.0，自适应范围 0.5–2.5 |
| 精度 | base 为 BF16，LoRA 参数由 PEFT 管理精度；非量化 |
| LoRA | r=16、alpha=32、dropout=0、bias=none |
| LoRA 目标层 | q/k/v/o_proj、gate/up/down_proj |
| 优化器 | PyTorch AdamW，weight_decay=0，固定学习率 |
| 梯度裁剪 | norm 1.0 |
| 保存频率 | 每 20 个 optimizer steps，以及最后一步 |

64/4/1024/2048/1e-6/300 对应附录 A 的主要设置；按 64 个 prompt、每题 4 个 rollout 解读全局批量。AdamW 细节、梯度裁剪和 token 平均为此实现的显式工程选择。实际逐条生成、逐条计算冻结评分、逐条反向以降低峰值显存，但直到完整 256 条回答反向完成才更新一次，因此不会在一批 rollout 中途改变学生参数。论文的短训练长度不等于评测生成长度。

## 运行示例

在项目根目录启动；将占位目录名换成你服务器 `/models` 下的实际目录：

```bash
CUDA_VISIBLE_DEVICES=0 python train/train_direct_opd.py \
  --student STUDENT_DIR \
  --teacher POST_RL_TEACHER_DIR \
  --teacher-ref PRE_RL_TEACHER_DIR
```

默认启用 LoRA，无需额外开关。可使用 `--lora-r 16 --lora-alpha 32` 或 `--lora-target-modules q_proj v_proj` 调整配置。目标层默认按 Qwen/Llama 风格命名，其他结构须填写实际模块名。学习率暂保留 1e-6，不代表这是已调优的 LoRA 学习率。

单进程显式分配模型到三张可见 GPU（不是 tensor parallel 或 DDP）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python train/train_direct_opd.py \
  --student STUDENT_DIR --teacher POST_RL_TEACHER_DIR --teacher-ref PRE_RL_TEACHER_DIR \
  --student-device cuda:0 --teacher-device cuda:1 \
  --teacher-ref-device cuda:2
```

可以把教师放到 CPU，例如 `--teacher-device cpu --teacher-ref-device cpu`，但 CPU BF16 运算是否高效取决于服务器，速度可能很慢。anchor 始终与学生共享设备。训练学生本身仍须放入一张 GPU，不支持自动模型切分、DDP、FSDP、ZeRO 或多进程启动。这版优先提供能审阅的单进程算法实现，不实现论文 8 卡训练系统的吞吐量。

共享 anchor 省去一份完整学生权重，LoRA 也将梯度和 Adam 状态限制在 adapter 参数上，但仍需学生及两个教师的权重、激活和 rollout KV cache；不承诺所有模型组合都能装入 A800 80GB。梯度检查点已开启，dropout 显式禁用。可把教师放到其他设备；调整 `--prompts-per-step` 改变有效训练批量，而不是逐条反向的峰值。LoRA 是在此前 top-1 简化之外新增的参数高效训练改动，不等同于论文原实验配置。

## 保存、恢复与评测

输出位于 `train/outputs/<UTC时间>/`：

- `config.json`：参数、实际包版本、数据过滤计数。
- `metrics.jsonl`：每步 teacher shift、policy loss、sampled KL、alpha、梯度范数、输出长度和耗时。
- `checkpoint-N/`：PEFT adapter 权重、adapter 配置与 tokenizer，以及 optimizer/alpha/step 状态；不重复保存 base model。保存完毕后才写 `complete.json`，不从不完整检查点恢复。

恢复时仍传入原始三个模型目录；`--student` 必须是最初学生，不要改成 checkpoint，否则 KL anchor 会改变：

```bash
python train/train_direct_opd.py \
  --student STUDENT_DIR --teacher POST_RL_TEACHER_DIR --teacher-ref PRE_RL_TEACHER_DIR \
  --resume train/outputs/RUN/checkpoint-100 --steps 300
```

恢复会校验核心配置（包括 LoRA 配置），允许改变设备、输出目录、保存间隔和最终 steps。旧版全参数检查点不能作为新版 LoRA 的 `--resume` 输入。以 seed+step 初始化每步随机数；相同设备/软件/数据下旨在保持 step 级采样可复现，不保证不同硬件逐 bit 一致。不要在恢复前更换原始模型或数据内容；路径相同并不保证文件内容未变。

现有评测要求完整模型，不能直接把 adapter 检查点作为 `--model`。先在单独进程离线合并，再评测：

```bash
python train/merge_lora.py \
  --checkpoint train/outputs/RUN/checkpoint-300 \
  --output train/outputs/RUN/merged-300
python eval/run_eval.py --model train/outputs/RUN/merged-300
```

合并默认在 CPU 上进行，需要足够主机内存；可用 `--device cuda:0` 改为 GPU。若搬迁了模型路径，用 `--base-model /新的原始模型路径` 指向相同的原始 base。合并导出到新目录，不修改原始 base 或 adapter；已存在的输出目录会报错。仅合并后的模型用于推理，继续训练须恢复未合并的 adapter 检查点。

评测现已统一要求 `Answer:`，评分器支持英文/中文冒号，同时兼容 boxed 输出。eval 将逐步推理与格式指令放在 system 消息，user 保留去掉已知后缀后的题目正文；本训练默认仍使用附录 A 的单条 user DAPO 模板，所以答案格式已统一，但消息角色布局并不完全相同。eval 使用 temperature=0.6、top_p=1.0、最多生成4096 tokens；论文附录 A 使用 temperature=0.7、top_p=0.95、最多生成31744 tokens，并用 avg@32。这些协议仍有差异，现有评测结果不能直接与论文表格比较。

## 兼容性与边界

迁移目标为 Python 3.12.3、accelerate 1.15.0、torch 2.10.0+cu128、Transformers 5.3.0、TRL 1.13.0；LoRA 补充依赖选为 PEFT 0.19.0。`train/requirements.txt` 引用根目录共有依赖，版本集中在 `constraints.txt`。训练/合并改用 Transformers 5 的 `dtype`，模板调用明确使用 TRL 的 `processing_class` 参数。共享 base、禁用 adapter 计算参考、LoRA 保存和合并流程保持不变。

训练脚本不导入或安装 vLLM。目标 vLLM 0.19.1 的官方依赖排除 Transformers 5.3.x，这属于评测环境的已知声明冲突，详见根目录 README。你可以在服务器自行运行 `python check_environment.py --profile train` 检查训练直接依赖；此脚本不加载模型。未安装新训练环境、未导入验证或运行训练，不能将静态迁移视为已通过实测。旧 adapter/optimizer 的跨框架版本恢复也尚未验证。

本实现检查三个 tokenizer 的完整 token-to-ID 映射及模型输出词表维度一致，不支持跨 tokenizer 对齐。不同家族即使都叫 Qwen，也可能因新增 token、特殊 token 或词表尺寸不同而失败；必须先选择同词表模型组，不能删除校验强行相减。相同映射之外也需自行确认 tokenizer 语义兼容。跨家族泛化、重新训练 RL 教师、论文各基线、消融曲线及多阶段政策组合不包含在这版核心训练脚本中。
