"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"`
```
"""

from dataclasses import dataclass
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
from torch.utils.data import default_collate
from lerobot.common.envs.task_prompts import make_env_task_prompt
import imageio
import numpy
import torch
from transformers import AutoProcessor

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.lightning import LerobotLightningWrapper
from lerobot.common.policies.normalize import Normalize
from lerobot.common.policies.vlm.configuration_vlm_policy import VLMPolicyConfig
from lerobot.configs.types import FeatureType


@dataclass
class VLMCollateFunction:
    processor: AutoProcessor
    config: VLMPolicyConfig
    normalize_inputs: Normalize
    num_action_dim: int

    # creates a collate function for the VLM policy which uses the Qwen2VL preprocessor
    # to generate a batch of data from the dataset
    def __call__(self, batch):
        # apply processor to the batch
        conversations = []
        images = []

        for data in batch:
            user_content = []
            # TODO: data is on CPU / normalisation constants are on CPU!
            data = self.normalize_inputs(data)

            if data["observation.image"].dim() == 3:
                data["observation.image"] = data["observation.image"].unsqueeze(0)

            user_content.append(
                {
                    "type": "text",
                    "text": f"Instruction: {data['task']}\n",
                }
            )

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

            action_chunk = []

            for i in range(self.config.chunk_size):
                for j in range(self.num_action_dim):
                    action_chunk.append("<|box_start|>")

                # we ignore the separator between actions (this will delimit the axis)
                action_chunk.append("<|box_end|>")

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
                        {
                            "type": "text",
                            "text": "".join(action_chunk),
                        },
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

        start_action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|box_start|>")
        end_action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|box_end|>")

        inputs_action_mask = (inputs.input_ids == start_action_token_id) | (
            inputs.input_ids == end_action_token_id
        )
        action_mask = inputs.input_ids == start_action_token_id
        inputs.update(
            {
                "inputs_action_mask": inputs_action_mask,
                "action_mask": action_mask,
            }
        )

        return inputs

def main():
    # Create a directory to store the video of the evaluation
    output_directory = Path("outputs/eval/example_pusht_vlm")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = "cuda"

    # Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
    pretrained_policy_path = (
        "/mnt/scratch/users/as2180/lerobot_vla/pusht_vla__lr=5e5-bs=128-lora_r=64-lora_alpha=128/lightning_logs/version_0/checkpoints/qwenvla-global_step=23999.0-train_loss=0.02.ckpt"
    )
    # TODO: how do we make this compatible with the original LeRobot policy?
    policy = LerobotLightningWrapper.load_from_checkpoint(pretrained_policy_path, map_location=device)

    # Initialize evaluation environment to render two observation types:
    # an image of the scene and state/position of the agent. The environment
    # also automatically stops running after 300 interactions/steps.
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=300,
    )

    # Reset the policy and environments to prepare for rollout
    policy.model.reset()

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    vlm_model_name = "Qwen/Qwen2-VL-7B-Instruct"
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
    cfg = policy.model.config
    normalize_inputs = policy.model.normalize_inputs.to("cpu")
    num_action_dim = env.action_space.shape[0]
    vlm_collate = VLMCollateFunction(vlm_processor, cfg, normalize_inputs, num_action_dim=num_action_dim)

    numpy_observation, info = env.reset(seed=42)

    # Prepare to collect every rewards and all the frames of the episode,
    # from initial state to final state.
    rewards = []
    frames = []

    # Render frame of the initial state
    frames.append(env.render())
    instruction = "Push the T-shaped block onto the T-shaped target."
    step = 0
    done = False

    while not done:
        # Prepare observation for the policy running in Pytorch
        state = torch.from_numpy(numpy_observation["agent_pos"]).unsqueeze(0)
        image = torch.from_numpy(numpy_observation["pixels"]).unsqueeze(0)
        state = state.to(torch.float32)

        batch = [{"observation.state": state, "observation.image": image, "task": instruction}]

        policy_inputs = vlm_collate(batch)

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            policy_inputs = {key: value.to(device) for key, value in policy_inputs.items()}
            action = policy.model.select_action(policy_inputs)

        # Prepare the action for the environment
        numpy_action = action.squeeze(0).to("cpu").numpy()

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reach (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure!")

    env.close()

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

    print(f"Video of the evaluation is available in '{video_path}'.")

if __name__ == "__main__":
    main()
