#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Optional

from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("vlm")
@dataclass
class VLMPolicyConfig(PreTrainedConfig, Qwen2VLConfig):
    # default LORA config from PEFT
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    max_action_dim: int = 32
    # Decoding
    num_diffusion_steps: int = 10
    # Number of hidden layers for the action adapter
    action_input_adapter_mlp_depth: int = 2

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training and loss computation.
    dropout: float = 0.1

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # VLM Config
    vlm_config: Optional[Qwen2VLConfig] = None

    def __post_init__(self):
        Qwen2VLConfig.__init__(self, **self.vlm_config.to_dict())
        return super().__post_init__()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.robot_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
