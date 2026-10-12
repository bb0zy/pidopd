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
    teacher_vllm_base_url: str = field(
        default="http://127.0.0.1:8001/v1",
        metadata={"help": "Base URL of the teacher's vLLM OpenAI-compatible server."},
    )
    teacher_vllm_model: str = field(
        default="teacher",
        metadata={"help": "Served model name used by the teacher API."},
    )
    teacher_vllm_api_key: str = field(
        default="EMPTY",
        metadata={"help": "API key for the teacher server."},
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

