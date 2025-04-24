from dataclasses import dataclass
from typing import AbstractSet, Any, Dict, Type

from torch import nn

from doomllm.logger import get_logger

logger = get_logger(__name__)


class _ModelRegistry:
    def __init__(self):
        self.models: Dict[str, Type[nn.Module]] = {}

    def get_supported_archs(self) -> AbstractSet[str]:
        return set(self.models.keys())

    def get_model(self, arch: str) -> Type[nn.Module]:
        try:
            return self.models[arch]
        except KeyError:
            raise ValueError(f"Unsupported architecture: {arch}")

    def register_model(self, arch: str, model: Type[nn.Module]):
        if arch in self.models:
            logger.warning(f"Overwriting existing model: {arch}")
        self.models[arch] = model


ModelRegistry = _ModelRegistry()
