from lightning import LightningModule
from torch.optim import AdamW


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
        return AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
        )
