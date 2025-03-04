from lightning import LightningModule
from lightning.fabric.utilities.cloud_io import _load as pl_load
from torch.optim import AdamW
from transformers import AutoConfig, AutoProcessor, get_scheduler

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.vlm.configuration_vlm_policy import VLMPolicyConfig
from lerobot.common.policies.vlm.modeling_vlm_policy import VLMPolicy
from lerobot.configs.types import FeatureType


class LerobotLightningWrapper(LightningModule):
    def __init__(self, model, **kwargs):
        super(LerobotLightningWrapper, self).__init__()
        self.model = model
        self.save_hyperparameters(kwargs)

    def forward(self, batch):
        return self.model.forward(batch)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss)
        return val_loss

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=None, **kwargs
    ):
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
        input_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.STATE}
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
        checkpoint = pl_load(checkpoint_path, map_location=map_location)
        wrapper = LerobotLightningWrapper(policy, **checkpoint["hyper_parameters"])
        wrapper.load_state_dict(checkpoint["state_dict"])
        return wrapper

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_scheduler(
            "cosine",
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def reset(self):
        self.model.reset()
