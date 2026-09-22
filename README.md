# pidopd

本地数学模型 avg@32 评测，以及 Direct-OPD top-1 LoRA 训练。默认本地模型 `/models`、本地数据 `dataset/`；模型与数据加载强制离线。保持现有训练算法、LoRA 共享 base 参考模型和 `Answer:` 评测格式。

## 新环境目标

| 包 | 目标版本 |
| --- | --- |
| accelerate | 1.15.0 |
| torch | 2.10.0+cu128 |
| transformers | 5.3.0 |
| trl | 1.13.0 |
| vllm | 0.19.1 |
| peft（LoRA 补充依赖） | 0.19.0 |

Python 未收到新的版本要求，沿用 3.12.3；服务器为 Linux/NVIDIA GPU。PEFT 0.19.0 是本项目为迁移选定的版本，不代表你已安装。上表由根目录 `constraints.txt` 统一约束。没有安装新环境、导入新框架、运行训练/评测或验证 CUDA ABI。

**存在一个需要区分的依赖冲突**：官方 vLLM 0.19.1 的 CUDA 依赖指定 torch 2.10.0，与目标 Torch 主版本一致；但其 `requirements/common.txt` 明确排除 Transformers 5.3.x。因此这五个包是用户指定的目标，而不是已经验证兼容的官方安装组合。若服务器使用定制 wheel，其依赖声明可能不同；仍需实际验证。不要通过忽略依赖错误来认定 ABI 或接口兼容。

源码与接口核对参考：

- [vLLM 0.19.1 CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/cuda.txt)
- [vLLM 0.19.1 Transformers 排除规则](https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/common.txt)
- [TRL 1.13.0 模板接口](https://github.com/huggingface/trl/blob/v1.13.0/trl/data_utils.py)
- [Transformers 5.3.0 模型接口](https://huggingface.co/docs/transformers/v5.3.0/en/main_classes/model)

## 依赖结构与检查

- `requirements.txt`：训练/评测共有依赖，不安装 vLLM 或 PEFT。
- `eval/requirements.txt`：Transformers 评测依赖。
- `train/requirements.txt`：共有依赖加 PEFT；不安装 vLLM。
- `eval/requirements-vllm.txt`：额外声明 vLLM 目标。官方包解析可能因上述冲突失败，不能将其当作可直接安装的 lockfile。

你已配置好的服务器无需盲目重装。可先自行运行只读检查（不导入 Torch、不开 GPU、不下载文件）：

```bash
python check_environment.py --profile train
python check_environment.py --profile vllm
python -m pip check
```

检查脚本读取已安装 wheel 元数据、对照目标版本并检查所选包的直接依赖；退出码 1 表示缺包、版本偏差或声明冲突。它不检查 GPU 驱动、CUDA 内核、所有传递依赖或运行时兼容性，也不自动修改环境。目标 torch wheel 是 `+cu128`，需要服务器已有对应 wheel/镜像源；依赖文件不会自动切换到 CPU 版本或下载模型。

若只能使用官方包，应选择彼此支持的版本后同步修改 `constraints.txt`；本次没有擅自替换你的五个目标版本。Transformers 后端和训练流程本身不导入 vLLM，可独立使用。

## 入口

```bash
# 默认 vLLM 全量 avg@32；依赖冲突需在服务器确认
python eval/run_eval.py --model /models/qwen --max-num-seqs 32

# Transformers 单卡评测，不导入 vLLM
python eval/run_eval.py --model /models/qwen --backend transformers

# LoRA 训练：三个本地模型目录由运行时指定
python train/train_direct_opd.py --student STUDENT_DIR --teacher POST_RL_DIR --teacher-ref PRE_RL_DIR

# 单独导出合并模型，供 eval 使用
python train/merge_lora.py --checkpoint train/outputs/RUN/checkpoint-300 --output train/outputs/RUN/merged-300
```

详见 [评测说明](eval/README.md) 和 [训练说明](train/README.md)。升级框架后，旧 adapter 应在服务器确认可加载；旧优化器状态的跨版本精确续训未验证。检查点格式及原始 base 参考语义没有改变。
