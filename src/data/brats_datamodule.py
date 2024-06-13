from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader
from monai.transforms import Compose
from lightning import LightningDataModule
from .components.preparing_data import get_volumes_path, BratsDataset
from .components.data_transforms import permute_and_add_axis_to_mask, spatialpad


class BratsDataModule(LightningDataModule):
    """LightningDataModule Implements 7 key methods"""
    def __init__(self,
                 data_dir: str = "data/", # data_directory. Defaults to "data/"
                 batch_size: int=1, # Batch size for the dataloaders, defaults to 1
                 num_workers: int=0, # Number of worker processes for data loading, defaults to 0
                 pin_memory: bool = False, # If True, the data loader will copy Tensors into CUDA pinned memory before returning them, defaults to False
                 ) -> None:
        
        super().__init__()

        # This allows to access init params with "self.hparams" attribute. also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir # Directory for data
        self.batch_size = batch_size 

        # data transformations
        self.transform = Compose([ # Custom transform function
            permute_and_add_axis_to_mask(),
            spatialpad(image_target_size=[4, 256, 256, 256], mask_target_size=[1, 256, 256, 256]),
        ])

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self) -> None: # can't make state assignments, only used for task like downloading dataset
        pass

    def setup(self, stage=None) -> None: # 'stage' arg is used to separate logic for 'fit' and 'test', runs across all GPUs and is safe to make state assignments
        # Get paths to data volumes and segmentations
        train_volumes_path, train_segmentations_path, val_volumes_path, val_segmentations_path, test_volumes_path, test_segmentations_path = get_volumes_path(self.data_dir)
        
        # Assign training and validation datasets for the 'fit' stage
        if stage == "fit" or stage is None:
            self.data_train = BratsDataset(train_volumes_path, train_segmentations_path, transform=self.transform)
            self.data_val = BratsDataset(val_volumes_path, val_segmentations_path, transform=self.transform)
        
        # Assign test dataset for the 'test' stage
        if stage == 'test' or stage is None:
            self.data_test = BratsDataset(test_volumes_path, test_segmentations_path, transform=self.transform)
        # self.data_predict =  BratsDataset(test_volumes_path,test_segmentations_path,transform=self.transform)

    def train_dataloader(self): # Method to create and return the training dataloader
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )
    
    # def predict_dataloader(self) -> Any:
    #     return DataLoader(datasset=self.data_predict,
    #                       batch_size=self.batch_size,
    #                       num_workers=self.hparams.num_workers,
    #                       pin_memory=self.hparams.pin_memory,
    #                       shuffle=False,
    #                       drop_last=True
    #                              )
    
    def teardown(self, stage): # Method to clean up after each stage
        pass



if __name__ == "__main__":
    _ = BratsDataModule()