from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule
from src.data.brats_datamodule import BratsDataModule


@pytest.mark.parametrize("batch_size",[2,])
def test_brats_datamodule(batch_size:int):
    data_dir = 'D:\\Neuroscience and Neuroimaging\\CAP5516 Medical Image Computing\\MSD\\brats_subset'

    dm.setup()
    dm = BratsDataModule(data_dir=data_dir)

    assert dm.train_dataloader() != None, f'Cannot create train dataloader'

    assert len(dm.train_dataloader()) != 0 , f'Train loader empty'


    assert dm.val_dataloader() != None, f'Cannot create val dataloder'
    assert len(dm.val_dataloader()) != 0, f'Val loader empty'
    

@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
