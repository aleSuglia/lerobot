"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from torch.utils.data import default_collate
from transformers import AutoConfig, AutoProcessor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.envs.task_prompts import make_env_task_prompt
from lerobot.common.policies.lightning import LerobotLightningWrapper
from lerobot.common.policies.normalize import Normalize
from lerobot.common.policies.vlm.configuration_vlm_policy import VLMPolicyConfig
from lerobot.common.policies.vlm.modeling_vlm_policy import VLMPolicy
from lerobot.configs.types import FeatureType


@dataclass
class VLMCollateFunction:
    processor: AutoProcessor
    config: VLMPolicyConfig
    normalize_inputs: Normalize

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
            data = self.normalize_inputs(data)

            if data["observation.image"].dim() == 3:
                data["observation.image"] = data["observation.image"].unsqueeze(0)

            for img, state in zip(data["observation.image"], data["observation.state"], strict=False):
                user_content.append({"type": "image"})
                rounded_state = [round(s, 2) for s in state.tolist()]

                user_content.append(
                    {
                        "type": "text",
                        "text": f"State: {rounded_state}\n",
                    },
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
                    "role": "system",
                    "content": [{"text": make_env_task_prompt("pusht")}],
                },
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
            self.processor.apply_chat_template(msg, add_generation_prompt=False, add_vision_id=True)
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


def main(args):
    # Create a directory to store the training checkpoint.
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Data loading
    num_workers = args.num_workers
    batch_size = args.batch_size

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    num_training_steps = args.num_training_steps
    num_warmup_steps = args.num_warmup_steps
    log_freq = args.log_freq
    num_action_chunk_size = args.num_action_chunk_size

    # Training stages
    is_stage_one_training = args.is_stage_one_training
    is_stage_two_training = args.is_stage_two_training

    if not is_stage_one_training and not is_stage_two_training:
        print("Please specify at least one training stage. Assuming stage one training.")
        is_stage_one_training = True

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.STATE}

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    vlm_model_name = "Qwen/Qwen2-VL-7B-Instruct"
    vlm_config = AutoConfig.from_pretrained(vlm_model_name)
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
    action_start_token_id = vlm_processor.tokenizer.convert_tokens_to_ids("<|box_start|>")

    vlm_config.update(
        {"num_hidden_layers": args.num_hidden_layers, "action_start_token_id": action_start_token_id}
    )
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
        model_config=cfg,
        dataset_stats=dataset_metadata.stats,
        learning_rate=args.learning_rate,
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
    normalize_inputs = Normalize(cfg.input_features, cfg.normalization_mapping, dataset_metadata.stats)

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=VLMCollateFunction(vlm_processor, cfg, normalize_inputs),
    )

    trainer = Trainer(max_steps=num_training_steps, default_root_dir=output_directory)

    trainer.fit(policy, train_dataloader)

    print("---Finished training---")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset", type=str, default="lerobot/pusht", help="Dataset name.")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="Number of hidden layers.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--num_training_steps", type=int, default=5, help="Number of training steps.")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of warmup steps.")
    parser.add_argument("--log_freq", type=int, default=1, help="Logging frequency.")
    parser.add_argument("--num_action_chunk_size", type=int, default=15, help="Number of action chunk size.")
    parser.add_argument("--is_stage_one_training", action="store_true", help="Stage one training.")
    parser.add_argument("--is_stage_two_training", action="store_true", help="Stage two training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument(
        "--output_directory", type=str, default="outputs/train/example_pusht_vlm", help="Output directory."
    )

    args = parser.parse_args()
    main(args)
