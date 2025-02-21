"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from torch.utils.data import default_collate
from transformers import AutoConfig, AutoProcessor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.lightning import LerobotLightningWrapper
from lerobot.common.policies.vlm.configuration_vlm_policy import VLMPolicyConfig
from lerobot.common.policies.vlm.modeling_vlm_policy import VLMPolicy
from lerobot.configs.types import FeatureType


@dataclass
class VLMCollateFunction:
    processor: AutoProcessor
    config: VLMPolicyConfig

    # creates a collate function for the VLM policy which uses the Qwen2VL preprocessor
    # to generate a batch of data from the dataset
    def __call__(self, batch):
        # apply processor to the batch
        conversations = []
        images = []
        actions = []

        for data in batch:
            actions.append(data["action"])
            user_content = []

            if data["observation.image"].dim() == 3:
                data["observation.image"] = data["observation.image"].unsqueeze(0)

            for img, state in zip(data["observation.image"], data["observation.state"], strict=False):
                user_content.append({"type": "image"})
                user_content.append(
                    {
                        "type": "text",
                        "text": f"State: {state.tolist()}\n",
                    }
                )
                images.append(img)

            user_content.append(
                {
                    "type": "text",
                    "text": f"Instruction: {data['task']}\n",
                }
            )

            conversation = [
                {
                    "role": "user",
                    "content": user_content,
                },
                # this represents the actual set of action tokens
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "".join(["<|box_start|>"] * self.config.chunk_size)},
                    ],
                },
            ]

            conversations.append(conversation)

        texts = [
            self.processor.apply_chat_template(msg, add_generation_prompt=True, add_vision_id=True)
            for msg in conversations
        ]
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        inputs.update({"action": default_collate(actions)})
        return inputs


def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/example_pusht_vlm")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Data loading
    num_workers = 0
    batch_size = 5

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    num_training_steps = 5000
    num_warmup_steps = 100
    log_freq = 1
    num_action_chunk_size = 15

    # Training stages
    is_stage_one_training = True
    is_stage_two_training = False

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    vlm_model_name = "Qwen/Qwen2-VL-7B-Instruct"
    vlm_config = AutoConfig.from_pretrained(vlm_model_name)
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
    action_start_token_id = vlm_processor.tokenizer.convert_tokens_to_ids("<|box_start|>")

    vlm_config.update({"num_hidden_layers": 2, "action_start_token_id": action_start_token_id})
    cfg = VLMPolicyConfig(
        vlm_config=vlm_config,
        input_features=input_features,
        output_features=output_features,
        chunk_size=num_action_chunk_size,
        is_stage_one_training=is_stage_one_training,
        is_stage_two_training=is_stage_two_training,
    )

    # We can now instantiate our policy with this config and the dataset stats.
    policy = VLMPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy = LerobotLightningWrapper(
        policy,
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    # In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
    delta_timestamps = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [0.0],
        "observation.state": [0.0],
        # Load current action and 14 future actions with a 0.1 seconds spacing.
        "action": np.arange(0, 0.1 * num_action_chunk_size, 0.1),
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=VLMCollateFunction(vlm_processor, cfg),
    )

    trainer = Trainer(max_steps=num_training_steps)

    trainer.fit(policy, dataloader)

    policy.model.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
