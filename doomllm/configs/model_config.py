from dataclasses import dataclass
from typing import Optional

from transformers import AutoConfig, AutoTokenizer

from doomllm.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    hf_config: AutoConfig
    revision: Optional[str] = None
    dtype: str = "auto"
    model_path: Optional[str] = None
    trust_remote_code: bool = True
