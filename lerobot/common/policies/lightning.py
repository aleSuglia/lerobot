from lightning import LightningModule
from torch.optim import AdamW
from transformers import get_scheduler


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

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
        )
        scheduler = get_scheduler(
            "cosine",
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def lr_scheduler_step(self, scheduler, metrics):
        scheduler.step(metrics=metrics, epoch=self.current_epoch)  # timm's scheduler need the epoch value
