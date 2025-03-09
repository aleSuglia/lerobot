from lightning import LightningModule
from lightning.fabric.utilities.cloud_io import _load as pl_load
from torch.optim import AdamW
from transformers import AutoConfig, AutoProcessor, get_scheduler

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.vlm.configuration_vlm_policy import VLMPolicyConfig
from lerobot.common.policies.vlm.modeling_vlm_policy import VLMPolicy
from lerobot.configs.types import FeatureType
from lightning.pytorch.core.saving import load_hparams_from_yaml

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
        self.log("train_loss", loss, on_step=True)
        self.log("global_step", self.global_step, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_step=True)
        return val_loss

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path, map_location=None, hparams_file=None, strict=None, **kwargs
    ):
        checkpoint = pl_load(checkpoint_path, map_location=map_location)
        # assert hparams_file is not None, "hparams.yaml file is always required!"
        # hparams = load_hparams_from_yaml(hparams_file)
        policy_config = checkpoint["hyper_parameters"]["model_config"]
        
        # We can now instantiate our policy with this config and the dataset stats.
        dataset_stats = checkpoint["hyper_parameters"]["dataset_stats"]
        policy = VLMPolicy(policy_config, dataset_stats=dataset_stats)
        wrapper = LerobotLightningWrapper(policy, model_config=policy_config)
        wrapper.load_state_dict(checkpoint["state_dict"])
        wrapper.to(map_location)
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
