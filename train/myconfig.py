"""Direct-OPD settings on top of the shared TRL training arguments."""

from dataclasses import dataclass

from trl import SFTConfig


@dataclass
class MyConfig(SFTConfig):
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    num_generations: int = 4
    loss_top_k: int = 16
    kl_beta: float = 1.0
    adaptive_kl: bool = True
    alpha_epsilon: float = 0.01
    alpha_min: float = 0.5
    alpha_max: float = 2.5
    student_model_path: str = "/models/qwen3-1.7b"
    model_before: str = "/models/DeepSeek-R1-Distill-Qwen-1.5B"
    model_after: str = "/models/JustRL-DeepSeek-1.5B"
    student_vllm_server_url: str = "http://127.0.0.1:8000"
    student_vllm_group_port: int = 51216
    teacher_gpu: str = "1"
    teacher_max_model_len: int = 8192
    teacher_score_chunk_size: int = 256
    disable_dropout: bool = True
    enable_thinking: bool = False
