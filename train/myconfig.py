from dataclasses import dataclass, field

from trl import SFTConfig


@dataclass
class MyConfig(SFTConfig):
   

    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    max_new_tokens: int = field(
        default=2048,
        metadata={"help": "Maximum number of tokens to generate per completion."},
    )
    loss_top_k: int = field(
        default=16,
        metadata={"help": "Number of student vocabulary candidates used per answer token in the loss."},
    )
    student_model_path: str = field(
        default="/models/qwen3-1.7b",
        metadata={"help": "Local model directory for the student and its tokenizer."},
    )
    student_vllm_server_url: str = field(
        default="http://127.0.0.1:8000",
        metadata={"help": "URL of the student vLLM server started with vllm/start_vllm.sh."},
    )
    student_vllm_group_port: int = field(
        default=51216,
        metadata={"help": "NCCL rendezvous port for student weight synchronization, distinct from the HTTP port."},
    )
    teacher_model_path: str = field(
        default="/models/qwen3-0.6b",
        metadata={"help": "Local model directory for the frozen Transformers teacher."},
    )
    teacher_gpu: str = field(
        default="1",
        metadata={"help": "CUDA_VISIBLE_DEVICES for the teacher process, independent of the student."},
    )
    teacher_max_model_len: int = field(default=8192)
    teacher_score_chunk_size: int = field(
        default=8192,
        metadata={"help": "Tokens per cached teacher forward; limits full-vocabulary logits memory."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )
    seq_kd: bool = field(
        default=False,
        metadata={
            "help": "Seq_kd parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised "
            "FT on teacher-generated output)."
        },
    )

