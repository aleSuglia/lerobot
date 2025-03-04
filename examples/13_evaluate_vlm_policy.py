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

    # creates a collate function for the VLM policy which uses the Qwen2VL preprocessor
    # to generate a batch of data from the dataset
    def __call__(self, batch):
        # apply processor to the batch
        conversations = []
        images = []

        for data in batch:
            user_content = []
            data = self.normalize_inputs(data)

            if data["observation.image"].dim() == 3:
                data["observation.image"] = data["observation.image"].unsqueeze(0)

            if data["observation.state"].dim() == 1:
                data["observation.state"] = data["observation.state"].unsqueeze(0)

            for img, state in zip(data["observation.image"], data["observation.state"], strict=False):
                user_content.append({"type": "image"})
                rounded_state = [round(s, 2) for s in state.tolist()]
                user_content.append(
                    {
                        "type": "text",
                        "text": f"State: {rounded_state}\n",
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
                # this represents the actual set of action tokens (they will be used to make the
                # forward pass)
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

        action_token_id = self.processor.tokenizer.encode("<|box_start|>")[0]
        x, y = torch.where(inputs["input_ids"] == action_token_id)

        inputs.update(
            {
                "prefix_input_ids": inputs["input_ids"][:, : y[0]],
                "prefix_attention_mask": inputs["attention_mask"][:, : y[0]],
            }
        )
        # we don't have any labels, so we just return the inputs
        # inputs.update({"action": default_collate(actions)})
        return inputs


def main():
    # Create a directory to store the video of the evaluation
    output_directory = Path("outputs/eval/example_pusht_vlm")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = "mps"

    # Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
    pretrained_policy_path = (
        "outputs/train/train/pusht_vlm_stage1/lightning_logs/version_0/checkpoints/epoch=12-step=5000.ckpt"
    )
    # OR a path to a local outputs/train folder.
    # pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

    # policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, map_location=device)
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

    # TODO: do we need this really?
    # # We can verify that the shapes of the features expected by the policy match the ones from the observations
    # # produced by the environment
    # print(policy.config.input_features)
    # print(env.observation_space)

    # # Similarly, we can check that the actions produced by the policy will match the actions expected by the
    # # environment
    # print(policy.config.output_features)
    # print(env.action_space)

    # Reset the policy and environments to prepare for rollout
    policy.model.reset()
    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.STATE}

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    vlm_model_name = "Qwen/Qwen2-VL-7B-Instruct"
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_name)
    cfg = policy.model.config
    normalize_inputs = Normalize(cfg.input_features, cfg.normalization_mapping, dataset_metadata.stats)
    vlm_collate = VLMCollateFunction(vlm_processor, cfg, normalize_inputs)

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
        state = torch.from_numpy(numpy_observation["agent_pos"])
        image = torch.from_numpy(numpy_observation["pixels"])

        state = state.to(torch.float32)

        batch = [{"observation.state": state, "observation.image": image, "task": instruction}]

        policy_inputs = vlm_collate(batch)

        # Predict the next action with respect to the current observation
        with torch.inference_mode():
            # TODO: then we pass the policy_inputs to the model; the rest must be the same
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

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

    print(f"Video of the evaluation is available in '{video_path}'.")


if __name__ == "__main__":
    main()
