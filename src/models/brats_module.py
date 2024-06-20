from typing import Any, Dict, Tuple

import torch
import torchio as tio
from lightning import LightningModule

from monai.metrics import CumulativeIterationMetric
from monai.losses import DiceLoss
from monai.metrics.meandice import DiceMetric


class BratsLitModule(LightningModule):
    """
    PyTorch Lightning module for training and evaluating models on the BRATS dataset.
    It implements 8 key methods of LightningModule
    """
    def __init__(self,
                 net: torch.nn.Module, # model to train
                #  optimizer: torch.optim.Optimizer, # optimizer to use for training
                #  scheduler: torch.optim.lr_scheduler, # learning rate scheduler to use for training
                #  compile: bool,
                 ) -> None:
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute, also ensure init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # Initialize dice loss
        self.dice_loss_fn = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        
        # Initialize dice score for calculating validation dice score and test dice score
        self.dice_score_fn_val = DiceMetric(include_background=False)
        self.dice_score_fn_test = DiceMetric(include_background=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model 'self.net'
        :param x: image of shape [BCDHW] -> [batch, 4, 155, 240, 240]
        :return: A tensor of logits of shape [BCDHW] -> [batch, 4, 155, 240, 240]
        """
        return self.net(X)

    def training_step(self,
                      batch,
                    #   batch: Dict[str, torch.Tensor ],
                      batch_idx: int):
        """Perform a single training step on a batch of data from training set.
        :param batch: A batch of data containing the input tensor of images and mask, accessed via key 'image' and 'mask'
        :param batch_idx: The index of the current batch
        :return: A tensor of losses between model predictions and targets
        """
        
        X, Y = batch['image'][tio.DATA], batch['mask'][tio.DATA]
        logits = self(X) # equivalent to self.forward(X)
        loss = self.dice_loss_fn(logits, Y) # logits of shape [batch, 4, 128, 240, 240] and loss_fn convertes Y to one-hot resulting the same shape as logits.
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # If logger=False, it (train_loss) won't be logged into csv file or TensorBoard depending upon logger=csv is used or something else.
        return loss

        # PL automatically does gradient cleaning{model.zero_grad()}, accumulating derivatives{loss.backward()}, and step in opposite direction of gradient i.e. backpropagation (optimizer.step())
    
    def validation_step(self,
                        batch,
                        # batch: Dict[str, torch.Tensor],
                        batch_idx: int):
        """Perform a single validation step on a batch of data from the validation set"""

        X, Y = batch['image'][tio.DATA], batch['mask'][tio.DATA]
        val_logits = self(X)
        predicted_class_labels_val = torch.argmax(val_logits, dim=1, keepdim=True) # val_logits of shape [batch, 4, D, H, W] {raw logits}, after argmax [batch, D, H, W] {0, 1, 2, 3}, since it takes argmax along the channels (or, the #classes)
        batch_dice_score_val = self.dice_score_fn_val(predicted_class_labels_val, Y)

    def test_step(self,
                  batch,
                #   batch: Dict[str, torch.Tensor],
                  batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set"""
        
        X, Y = batch['image'][tio.DATA], batch['mask'][tio.DATA]
        test_logits = self(X)
        predicted_class_labels_test = torch.argmax(test_logits, dim=1, keepdim=True)
        batch_dice_score_test = self.dice_score_fn_test(predicted_class_labels_test, Y)
        # pass

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends"""
        val_avg_dice = self.dice_score_fn_val.aggregate().item()
        self.log('val_score', val_avg_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.dice_score_fn_val.reset()

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends"""
        test_avg_dice = self.dice_score_fn_test.aggregate().item()
        self.log('test_score', test_avg_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.dice_score_fn_test.reset()

    def configure_optimizers(self):
        """Configure what optimizers and learning-rate schedulers to use in our optimization"""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    

if __name__ == "__main__":
    _ = BratsLitModule(None, None, None, None)