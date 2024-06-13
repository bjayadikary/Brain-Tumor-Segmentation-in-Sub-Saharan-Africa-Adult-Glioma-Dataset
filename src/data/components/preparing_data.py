import os
from glob import glob as glob
import nibabel as nib
import numpy as np

from torch.utils.data import Dataset


def get_volumes_path(base_data_dir):
    train_volumes_path = sorted(glob(os.path.join(base_data_dir, 'TrainVolumes', '*.nii.gz')))
    train_segmentations_path = sorted(glob(os.path.join(base_data_dir, 'TrainSegmentation', '*.nii.gz')))
    
    val_volumes_path = sorted(glob(os.path.join(base_data_dir, 'ValVolumes', '*.nii.gz')))
    val_segmentations_path = sorted(glob(os.path.join(base_data_dir, 'ValSegmentation', '*.nii.gz')))

    test_volumes_path = sorted(glob(os.path.join(base_data_dir, 'TestVolumes', '*.nii.gz')))
    test_segmentations_path = sorted(glob(os.path.join(base_data_dir, 'TestSegmentation', '*.nii.gz')))

    return train_volumes_path, train_segmentations_path, val_volumes_path, val_segmentations_path, test_volumes_path, test_segmentations_path


class BratsDataset(Dataset):
    def __init__(self, images_path_list, masks_path_list, transform=None):
        """
        Args:
            images_path_list (list of strings): List of paths to input images.
            masks_path_list (list of strings): List of paths to masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_path_list = images_path_list
        self.masks_path_list = masks_path_list
        self.transform = transform
        self.length = len(images_path_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load image
        image_path = self.images_path_list[idx]
        image = nib.load(image_path).get_fdata()
        image = np.float32(image) # shape of image [240, 240, 155, 4]

        # Load mask
        mask_path = self.masks_path_list[idx]
        mask = nib.load(mask_path).get_fdata()
        mask = np.float32(mask) # shape of mask [240, 240, 155]

        if self.transform:
            transformed_sample = self.transform({'image': image, 'mask': mask})
        
        return transformed_sample
