from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule
from src.data.brats_datamodule import BratsDataModule

from src.models.components.unet import UNet
from src.models.brats_module import BratsLitModule
from src.models.brats_module_for_generating_predictions import BratsLitModuleG

from src.models.components.mednext.MedNeXt import MedNeXt

from src.models.components.mednext.MedNeXt_with_linear_adapter import MedNeXtWithLinearAdapters
from src.models.components.mednext.MedNeXt_with_conv_adapter import MedNeXtWithAdapters
from collections import OrderedDict

import os

import torchio as tio



@pytest.mark.parametrize("batch_size",[1,])
def test_brats_datamodule(batch_size:int):
    data_dir = 'C:\\Users\\lenovo\\BraTS2023_SSA_modified_structure\\stacked_subset'

    dm:BratsDataModule = BratsDataModule(data_dir=data_dir)
    dm.setup()


    assert dm.train_dataloader() != None, f'Cannot create train dataloader'

    assert len(dm.train_dataloader()) != 0 , f'Train loader empty'


    assert dm.val_dataloader() != None, f'Cannot create val dataloder'
    assert len(dm.val_dataloader()) != 0, f'Val loader empty'

    train_batch = next(iter(dm.train_dataloader()))
    train_x, train_y = train_batch['image'][tio.DATA], train_batch['mask'][tio.DATA]
    
    assert train_x.dtype == torch.float32
    assert train_y.dtype == torch.float32

    val_batch = next(iter(dm.val_dataloader()))
    val_x, val_y = val_batch['image'][tio.DATA], val_batch['mask'][tio.DATA]

    assert val_x is not None, 'X is None'
    assert val_y is not None, 'Y is None'
    assert val_x.dtype == torch.float32
    assert val_y.dtype == torch.float32    

    # test if we can move data to gpu
    dm.transfer_batch_to_device(train_batch,'cuda', 0)

# This test function ensures that 'BratsDataModule' class initializes correctly with different batch sizes and the attributes are set as expected
@pytest.mark.parametrize("batch_size", [1, 2, 4]) # test will be run three times with 'batch_size' values of 1, 2, and 4
def test_brats_datamodule_initialization(batch_size: int): # takes 'batch_size' as an argument. This argument will take on each of the values specified in the parametrize decorator one by one
    data_dir = 'C:\\Users\\lenovo\\BraTS2023_SSA_modified_structure\\stacked_subset'
    dm = BratsDataModule(data_dir, batch_size=batch_size)
    assert dm.data_dir == data_dir # checks that 'data_dir' attribute of the 'BratsDataModule' instance is correctly set to the value specified during initialization
    assert dm.batch_size == batch_size # verify if 'batch_size' attribute of the 'BratsDataModule' instance is correclty set to the value specified during initialization


@pytest.fixture
def brats_datamodule(batch_size):
    return BratsDataModule(data_dir='C:\\Users\\lenovo\\BraTS2023_SSA_modified_structure\\stacked_subset', batch_size=batch_size)

# Test function to checks shapes and intensities of image and segmentation mask after applying the Transformations
# pytest will automatically call the brats_datamodule() fixture function before each test function that uses the brats_datamodule fixture as an argument. This means each test function will have access to a fresh instance of the BratsDataModule class, and can access its attributes and methods, such as train_transform
@pytest.mark.parametrize("batch_size", [1, 2])
def test_brats_datamodule_post_transform_shapes_and_intensities(brats_datamodule: BratsDataModule, batch_size:int):
    dm = brats_datamodule
    dm.setup()
    
    # Access the data from the datamodule
    train_data = dm.data_train
    # val_data = dm.data_val
    # test_data = dm.data_test

    # Test the train transform
    for subject in train_data:
        # Apply the transform on a subject
        transformed_subject = dm.train_transform(subject)
        transformed_subject_image_data = transformed_subject['image'][tio.DATA]
        transformed_subject_mask_data = transformed_subject['mask'][tio.DATA]

        # expected_image_shape = (4, 256, 256, 256) # Before the dataloader is called, we would have this shape. But if the data is coming from dataloder, the first dimension represent the batch dimension so the expected shape will be (1, 4, 256, 256, 256)
        expected_image_shape = (4, 128, 128, 128)
        assert transformed_subject_image_data.shape == expected_image_shape
        
        # Check if the mean of the transformed image is close to 0 and SD close to 1.
        transformed_mean = transformed_subject_image_data.mean().item()
        transformed_std = transformed_subject_image_data.std().item()
        # Assert mean of the transformed image is ~0 and std is ~1
        assert torch.isclose(torch.tensor(transformed_mean), torch.tensor(0.0), atol=1e-6), f'Mean is not ~0 after normalization {transformed_mean}'
        assert torch.isclose(torch.tensor(transformed_std), torch.tensor(1.0), atol=1e-6), f'Standard Deviation is not ~1 after normalization {transformed_std}'

        # expected_mask_shape = (1, 256, 256, 256)
        expected_mask_shape = (1, 128, 128, 128)
        assert transformed_subject_mask_data.shape == expected_mask_shape

        # Ensure the mask intensities are not transformed i.e. belongs to {0.0, 1.0, 2.0, 3.0}
        assert transformed_subject_mask_data.unique().tolist() == [0.0, 1.0, 2.0, 3.0]
        break


# This test funtion checks the 'BratsDataModule' class to ensure that it correctly initializes the data loaders ('train_dataloader', 'val_dataloader', and 'test_dataloader') with different batch sizes.
@pytest.mark.parametrize("batch_size", [1, 2])
def test_brats_datamodule_dataloaders(brats_datamodule: BratsDataModule, batch_size: int):
    dm = brats_datamodule

    dm.setup(stage='fit') # calls the 'setup' method of 'BratsDataModule' with 'stage=fit', preparign the data for training and validaiton stages.
    train_loader = dm.train_dataloader() # returns training data loader

    assert train_loader is not None, 'Cannot create train data loader'
    assert len(train_loader) > 0 # ensures 'train_loader' is not empty

    train_batch = next(iter(train_loader))
    train_x, train_y = train_batch['image'][tio.DATA], train_batch['mask'][tio.DATA]

    # Assert the data types are torch.float32
    assert train_x.dtype == torch.float32
    assert train_y.dtype == torch.float32

    # Assert the datas shapes coming from dataloader is valid
    expected_image_shape_from_dataloader = (batch_size, 4, 128, 128, 128)
    expected_mask_shape_from_dataloader = (batch_size, 1, 128, 128, 128)
    assert train_x.shape == expected_image_shape_from_dataloader
    assert train_y.shape == expected_mask_shape_from_dataloader

    # Similarly do for val and test dataloaders
    val_loader = dm.val_dataloader()
    assert val_loader is not None, 'Cannot create val data loader'
    assert len(val_loader) > 0, 'Val loader is Empty'
    val_batch = next(iter(val_loader))
    val_x, val_y = val_batch['image'][tio.DATA], val_batch['mask'][tio.DATA]
    assert val_x.dtype == torch.float32
    assert val_y.dtype == torch.float32
    assert val_x.shape == expected_image_shape_from_dataloader
    assert val_y.shape == expected_mask_shape_from_dataloader

    dm.setup(stage='test')
    test_loader = dm.test_dataloader()
    assert test_loader is not None, 'Cannot create test data loader'
    assert len(test_loader) > 0, 'Test laoder is empty'
    test_batch = next(iter(test_loader))
    test_x, test_y = test_batch['image'][tio.DATA], test_batch['mask'][tio.DATA]
    assert test_x.dtype == torch.float32
    assert test_y.dtype == torch.float32
    assert test_x.shape == expected_image_shape_from_dataloader
    assert test_y.shape == expected_mask_shape_from_dataloader



# This test function checks if the function resposible for transferring data to the GPU is valid or not
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_brats_datamodule_transfer_to_device(batch_size: int):
    data_dir = 'C:\\Users\\lenovo\\BraTS2023_SSA_modified_structure\\stacked_subset'
    dm = BratsDataModule(data_dir, batch_size=batch_size)
    dm.setup(stage='fit') # calls the 'setup' method of 'BratsDataModule' with 'stage=fit', preparign the data for training and validaiton stages.

    train_loader = dm.train_dataloader()
    train_batch = next(iter(train_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batch = dm.transfer_batch_to_device(train_batch, device, 0)
    train_x, train_y = train_batch['image'][tio.DATA], train_batch['mask'][tio.DATA]

    assert train_x.device.type == device.type
    assert train_y.device.type == device.type


# Test-cases for BratsLitModule
@pytest.fixture
def unet_model():
    model = UNet(
        spatial_dims=3, # 3 for using 3D ConvNet and 3D Maxpooling
        in_channels=4, # since 4 modalities
        out_channels=4, # 4 sub-regions to segment
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    )
    return model


def test_brats_litmodule_initialization(unet_model: torch.nn.Module):
    # Initialize optimizer and scheduler instances
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    module = BratsLitModule(net=unet_model, optimizer=optimizer, scheduler=scheduler)
    assert module.net == unet_model
    assert module.hparams is not None

@pytest.mark.parametrize("batch_size", [1, 2])
def test_brats_litmodule_forward_pass_unet(unet_model: torch.nn.Module, batch_size:int):
    # Initialize optimizer and scheduler instances
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    module = BratsLitModule(net=unet_model, optimizer=optimizer, scheduler=scheduler)
    input_tensor = torch.randn(batch_size, 4, 128, 128, 128)
    output = module.forward(input_tensor) # logits of shape (BCDHW)

    # Ensure logits is a tensor
    assert isinstance(output, torch.Tensor)
        
    expected_shape = (batch_size, 4, 128, 128, 128)
    assert output.shape == expected_shape


@pytest.mark.parametrize("batch_size", [1,])
def test_brats_litmodule_training_and_validation_step(unet_model: UNet, brats_datamodule: BratsDataModule, batch_size:int, capsys):
    dm = brats_datamodule
    dm.setup(stage='fit')
    train_loader = dm.train_dataloader()
    
    assert train_loader is not None
    assert len(train_loader) > 0

    train_batch = next(iter(train_loader))

    # Initialize optimizer and scheduler instances
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    
    module = BratsLitModule(net=unet_model, optimizer=optimizer, scheduler=scheduler)
    batch_idx = 0
    loss = module.training_step(train_batch, batch_idx)
    with capsys.disabled():
        print('train_batch size.....', len(train_batch))
        print('loss.......:', loss)
    assert isinstance(loss, torch.Tensor), f"Expected loss to be a torch.Tensor but got {type(loss)}"

    val_loader = dm.val_dataloader()
    assert val_loader is not None
    assert len(val_loader) > 0
    val_batch = next(iter(val_loader))

    with capsys.disabled():
        sample_dice_list, val_avg_dice = module.validation_step(val_batch, batch_idx) # to test this line change the validation_step to return sample_wise dice score and aggregated(batch_wise dice score) 
        if batch_size == 1:
            assert sample_dice_list.item() == val_avg_dice
            print('sample_dice_list and val_avg_dice: ', sample_dice_list, val_avg_dice)

    
    # assert module.dice_score_fn_val.aggregate() is not None



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


@pytest.fixture
def mednext_model(): # custom model
    model = MedNeXt(
        in_channels=4,
        n_classes=4,
        n_channels=4,
        kernel_size=3
    )
    return model

@pytest.mark.parametrize("batch_size", [1,])
def test_brats_litmodule_forward_pass_mednext(mednext_model: torch.nn.Module, batch_size:int):
    # Initialize optimizer and scheduler instances
    optimizer = torch.optim.AdamW(mednext_model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    module = BratsLitModule(net=mednext_model, optimizer=optimizer, scheduler=scheduler)
    input_tensor = torch.randn(batch_size, 4, 128, 128, 128)
    output = module.forward(input_tensor) # logits of shape (BCDHW)

    # Ensure logits is a tensor
    assert isinstance(output, torch.Tensor)
        
    expected_shape = (batch_size, 4, 128, 128, 128)
    assert output.shape == expected_shape


########################
### Testing Checkpoints
########################
@pytest.mark.parametrize("batch_size", [1,])
def test_custom_checkpoint_test(batch_size:int, capsys):
    # Load the checkpoint
    checkpoint_path = "D:\\BrainHack\\hydra\\spark_himalaya_git\\logs\\train\\runs\\2024-07-03_15-16-21\\checkpoints\\best-checkpoint.ckpt"
    checkpoint = torch.load(checkpoint_path)

    with capsys.disabled():
        for key in checkpoint.keys():
            print(key)


@pytest.mark.parametrize("batch_size", [1,])
def test_brats21_checkpoint_test(batch_size:int, capsys): # The capsys.disabled() context manager allows the print statements to be displayed even when capturing is enabled.
    # Load the checkpoint
    checkpoint_path = "D:\\BrainHack\\hydra\\spark_himalaya_git\\logs\\train\\runs\\2024-07-03_15-16-21\\checkpoints\\best-checkpoint.ckpt"
    checkpoint = torch.load(checkpoint_path)

    with capsys.disabled():
        for key in checkpoint.keys():
            print(key)


@pytest.fixture
def mednext_model_with_adapters(): # custom model
    model = MedNeXtWithLinearAdapters(
        in_channels=4,
        n_classes=4,
        n_channels=4,
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2,2,2,2,2,2,2,2,2],
        adapter_dim_ratio=0.25,
    )
    return model

@pytest.mark.parametrize("batch_size", [1,])
def test_checkpoint_load_for_finetuning(mednext_model_with_adapters: torch.nn.Module, batch_size:int, capsys):
    # Initialize optimizer and scheduler instances
    optimizer = torch.optim.AdamW(mednext_model_with_adapters.parameters(), lr=0.002, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Load the pretrained model checkpoint
    pretrained_ckpt_path = "D:\\BrainHack\\hydra\\spark_himalaya_git\\logs\\train\\runs\\2024-07-12_13-26-22\\checkpoints\\best-checkpoint.ckpt"
    pretrained_checkpoint = torch.load(pretrained_ckpt_path)

    # Initialize the new model with adapters
    model_with_adapter = BratsLitModule(net=mednext_model_with_adapters, optimizer=optimizer, scheduler=scheduler)

    # Changing the keys in pretrained_checkpoint to match the keys in model_with_adapter_state_dict
    for layer_name in list(pretrained_checkpoint['state_dict'].keys()):
        if 'dec_block' in layer_name or 'bottleneck' in layer_name or 'enc_block' in layer_name:
            layer_name_split = layer_name.split(".")
            layer_name_split.insert(2, "0")
            new_layer_name = ".".join(layer_name_split)

            # deletes the respective key, returns the associated value of that old key
            layer_values = pretrained_checkpoint['state_dict'].pop(layer_name)
            
            # assign the returned value to the new_layer_name
            pretrained_checkpoint['state_dict'][new_layer_name] = layer_values
        else:
            pass
           
    # assert if the len in new_layer_name (keys) in pretrained_checkpoint['state_dict'] matches those in model_with_adapter_state_dict, excpet the fully connected adapter's weights and biases
    assert len(list(pretrained_checkpoint['state_dict'].keys())) == len(list(x for x in model_with_adapter.state_dict().keys() if 'fc' not in x))
    
    # assert if every key on pretrained_checkpoint state dict is in model_with_adapter_state_dict
    count = 0
    for modified_key in pretrained_checkpoint['state_dict'].keys():
        if modified_key not in model_with_adapter.state_dict().keys():
            count +=1
    assert count == 0

    # Load weights/biases (i.e. state_dict) from pretrained checkpoint to new model with adapter that match the layer names. strict=False ignores non-matching keys, allowing the model to load even if the state dictionary does not perfectly match the model.
    model_with_adapter.load_state_dict(pretrained_checkpoint['state_dict'], strict=False)

    # Freeze all layers except the adapter layers
    for name, param in model_with_adapter.named_parameters():
        if 'dice_loss_fn' in name:
            with capsys.disabled():
                print(name)
        if 'fc' not in name:
            param.requires_grad = False

    # verify which parameters are trainable
    # with capsys.disabled():
    #     for name, param in model_with_adapter.named_parameters():
    #         print(f"{name}: requires_grad={param.requires_grad}")

    # with capsys.disabled():
    #     # print(pretrained_checkpoint['optimizer_states'])
    #     # for key in pretrained_checkpoint.keys():
    #     #     print(key)
    #     print('Pretrained dict keys:')
    #     for name in pretrained_state_dict.keys():
    #         print(name)

    #     print('Adapter model dict keys')
    #     for name in new_state_dict.keys():
    #         print(name)


@pytest.fixture
def mednextv1_small_model_with_adapters(): # mednext_smallv1 model with linear adapter
    model = MedNeXtWithLinearAdapters(
        in_channels=4,
        n_classes=4,
        n_channels=32,
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2,2,2,2,2,2,2,2,2],
        adapter_dim_ratio=0.25,
    )
    return model

@pytest.mark.parametrize("batch_size", [1,])
def test_brats21_checkpoint_load_for_finetuning(mednextv1_small_model_with_adapters: torch.nn.Module, batch_size:int, capsys):
    # Initialize optimizer and scheduler instances
    optimizer = torch.optim.AdamW(mednextv1_small_model_with_adapters.parameters(), lr=0.002, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Load the pretrained model checkpoint
    pretrained_ckpt_path = "C:\\Users\\lenovo\\Desktop\\logs_from_server\\logs_from_naami_server\\jolly-armadillo-5\\best-checkpoint_spark_himal.ckpt"
    pretrained_checkpoint = torch.load(pretrained_ckpt_path)

    # Initialize the new model with adapters
    model_with_adapter = BratsLitModule(net=mednextv1_small_model_with_adapters, optimizer=optimizer, scheduler=scheduler)

    # with capsys.disabled():
    #     print(model_with_adapter.state_dict().keys())

    # Changing the keys in pretrained_checkpoint to match the keys in model_with_adapter_state_dict
    for layer_name in list(pretrained_checkpoint['state_dict'].keys()):
        if 'dec_block' in layer_name or 'bottleneck' in layer_name or 'enc_block' in layer_name:
            layer_name_split = layer_name.split(".")
            layer_name_split.insert(2, "0")
            new_layer_name = ".".join(layer_name_split)

            # deletes the respective key, returns the associated value of that old key
            layer_values = pretrained_checkpoint['state_dict'].pop(layer_name)
            
            # assign the returned value to the new_layer_name
            pretrained_checkpoint['state_dict'][new_layer_name] = layer_values
        else:
            pass
           
    # assert if the len in new_layer_name (keys) in pretrained_checkpoint['state_dict'] matches those in model_with_adapter_state_dict, excpet the fully connected adapter's weights and biases
    assert len(list(pretrained_checkpoint['state_dict'].keys())) == len(list(x for x in model_with_adapter.state_dict().keys() if 'fc' not in x and 'dice_loss' not in x))
    
    # assert if every key on pretrained_checkpoint state dict is in model_with_adapter_state_dict
    count = 0
    for modified_key in pretrained_checkpoint['state_dict'].keys():
        if modified_key not in model_with_adapter.state_dict().keys():
            count +=1
    assert count == 0

    # Load weights/biases (i.e. state_dict) from pretrained checkpoint to new model with adapter that match the layer names. strict=False ignores non-matching keys, allowing the model to load even if the state dictionary does not perfectly match the model.
    model_with_adapter.load_state_dict(pretrained_checkpoint['state_dict'], strict=False)

    # Freeze all layers except the adapter layers
    for name, param in model_with_adapter.named_parameters():
        if 'dice_loss_fn' in name:
            with capsys.disabled():
                print(name)
        if 'fc' not in name:
            param.requires_grad = False


@pytest.fixture
def mednextv1_small_model_with_mednext_as_adapters(): # mednext_smallv1 model uses prefixed mednext block as adapter
    model = MedNeXtWithAdapters(
        in_channels=4,
        n_classes=4,
        n_channels=32,
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2,2,2,2,2,2,2,2,2],
    )
    return model

@pytest.mark.parametrize("batch_size", [1,])
def test_brats21_checkpoint_load_for_finetuning_with_mednext_as_adapter(mednextv1_small_model_with_mednext_as_adapters: torch.nn.Module, batch_size:int, capsys):
    # Initialize optimizer and scheduler instances
    optimizer = torch.optim.AdamW(mednextv1_small_model_with_mednext_as_adapters.parameters(), lr=0.002, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Load the pretrained model checkpoint
    pretrained_ckpt_path = "C:\\Users\\lenovo\\Desktop\\logs_from_server\\logs_from_naami_server\\jolly-armadillo-5\\best-checkpoint_spark_himal.ckpt"
    pretrained_checkpoint = torch.load(pretrained_ckpt_path)

    # Initialize the new model with adapters
    model_with_adapter = BratsLitModule(net=mednextv1_small_model_with_mednext_as_adapters, optimizer=optimizer, scheduler=scheduler)

    # with capsys.disabled():
    #     # for layer in model_with_adapter.state_dict().keys():
    #     #     print(layer)
    #     print(model_with_adapter.state_dict().keys())

    # Changing the keys in pretrained_checkpoint to match the keys in model_with_adapter_state_dict
    for layer_name in list(pretrained_checkpoint['state_dict'].keys()):
        if ('dec_block' in layer_name or 'bottleneck' in layer_name or 'enc_block' in layer_name) and 'med_adp' not in layer_name:
            layer_name_split = layer_name.split(".")
            layer_name_split.insert(2, "0")
            new_layer_name = ".".join(layer_name_split)

            # deletes the respective key, returns the associated value of that old key
            layer_values = pretrained_checkpoint['state_dict'].pop(layer_name)
            
            # assign the returned value to the new_layer_name
            pretrained_checkpoint['state_dict'][new_layer_name] = layer_values
        else:
            pass
           
    # assert if the len in new_layer_name (keys) in pretrained_checkpoint['state_dict'] matches those in model_with_adapter_state_dict, excpet the adapter's weights and biases, and also not including the 'dice_loss.weight' layer
    assert len(list(pretrained_checkpoint['state_dict'].keys())) == len(list(x for x in model_with_adapter.state_dict().keys() if 'med_adp' not in x and 'dice_loss' not in x))
    
    # assert if every key on pretrained_checkpoint state dict is in model_with_adapter_state_dict
    count = 0
    for modified_key in pretrained_checkpoint['state_dict'].keys():
        if modified_key not in model_with_adapter.state_dict().keys():
            count +=1
    assert count == 0

    # Load weights/biases (i.e. state_dict) from pretrained checkpoint to new model with adapter that match the layer names. strict=False ignores non-matching keys, allowing the model to load even if the state dictionary does not perfectly match the model.
    model_with_adapter.load_state_dict(pretrained_checkpoint['state_dict'], strict=False)

    # Freeze all layers except the adapter layers
    for name, param in model_with_adapter.named_parameters():
        if 'dice_loss_fn' in name:
            with capsys.disabled():
                print(name)
        if 'med_adp' not in name:
            param.requires_grad = False

#####################################
#### For generating prediction files
#####################################

@pytest.fixture
def ssa_datamodule_full(batch_size):
    return BratsDataModule(data_dir='C:\\Users\\lenovo\\BraTS2023_SSA_modified_structure\\utilities\\stacked', batch_size=batch_size)
    # return BratsDataModule(data_dir='C:\\Users\\lenovo\\BraTS2023_SSA_modified_structure\\stacked_subset', batch_size=batch_size)


@pytest.fixture
def mednextv1_small_model_with_small_linear_adapters(): # mednext_smallv1 model uses prefixed mednext block as adapter
    model = MedNeXtWithLinearAdapters(
        in_channels=4,
        n_classes=4,
        n_channels=32,
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2,2,2,2,2,2,2,2,2],
        adapter_dim_ratio=0.25
    )
    return model

@pytest.fixture
def mednextv1_small_model_with_large_linear_adapters(): # mednext_smallv1 model uses prefixed mednext block as adapter
    model = MedNeXtWithLinearAdapters(
        in_channels=4,
        n_classes=4,
        n_channels=32,
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2,2,2,2,2,2,2,2,2],
        adapter_dim_ratio=2
    )

    return model

@pytest.fixture
def mednextv1_small_model_with_conv_adapters(): # with convolution adapter
    model = MedNeXtWithAdapters(
        in_channels=4,
        n_classes=4,
        n_channels=32,
        exp_r=2,
        kernel_size=3,
        deep_supervision=False,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2,2,2,2,2,2,2,2,2],
    )

    return model

# @pytest.mark.parametrize("batch_size", [1,])
# def test_generate_prediction_files_with_large_linear_adapter(mednextv1_small_model_with_large_linear_adapters: torch.nn.Module, batch_size:int):
#     # Initialize optimizer and scheduler instances
#     optimizer = torch.optim.AdamW(mednextv1_small_model_with_large_linear_adapters.parameters(), lr=0.002, weight_decay=0.001)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

#     # Load the pretrained model checkpoint
#     pretrained_ckpt_path = ""
#     pretrained_checkpoint = torch.load(pretrained_ckpt_path)

#     module = BratsLitModule(net=mednextv1_small_model_with_mednext_as_adapters, optimizer=optimizer, scheduler=scheduler)
#     input_tensor = torch.randn(batch_size, 4, 128, 128, 128)
#     output = module.forward(input_tensor) # logits of shape (BCDHW)

#     # Ensure logits is a tensor
#     assert isinstance(output, torch.Tensor)
        
#     expected_shape = (batch_size, 4, 128, 128, 128)
#     assert output.shape == expected_shape



@pytest.mark.parametrize("batch_size", [1,])
def test_generate_prediction_files_with_conv_adapter(mednextv1_small_model_with_conv_adapters: torch.nn.Module, ssa_datamodule_full:BratsDataModule, batch_size:int, capsys):
    # Load the finetuned model checkpoint
    finetuned_ckpt_path = "C:\\Users\\lenovo\\Desktop\\logs_from_server\\logs_from_cc\\with_conv_adapter\\runs\\2024-07-20_04-08-04\\checkpoints\\best-checkpoint.ckpt"
    finetuned_checkpoint = torch.load(finetuned_ckpt_path)

    # Initialize optimizer and scheduler instances
    optimizer = torch.optim.AdamW(mednextv1_small_model_with_conv_adapters.parameters(), lr=0.002, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    module_with_conv_adapter = BratsLitModuleG(net=mednextv1_small_model_with_conv_adapters, optimizer=optimizer, scheduler=scheduler)

    # Assert if the number of keys in finetuned_checkpoint and keys in model with conv adapter is same 
    assert len(list(finetuned_checkpoint['state_dict'].keys())) == len(list(module_with_conv_adapter.state_dict().keys()))
    
    # Checking if layer names in both matches 
    with capsys.disabled():
        for finetuned_layer_name, layer_name in zip(list(finetuned_checkpoint['state_dict'].keys()), list(module_with_conv_adapter.state_dict().keys())):
            assert finetuned_layer_name == layer_name

    # Load the finetuned_checkpoint parameters in module_with_conv_adapter
    module_with_conv_adapter.load_state_dict(finetuned_checkpoint['state_dict'], strict=True)

    # Get the data modules
    dm = ssa_datamodule_full
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()

    # if batch_size == 1:
    #     assert len(test_loader) == 15

    # Loop through all the batch/samples
    for batch_idx, batch in enumerate(test_loader):
        # pass the test batch to the testing_step
        with capsys.disabled():
            test_avg_dice, mask_path = module_with_conv_adapter.test_step(batch, batch_idx)
            print(f'Dice Score on {os.path.basename(mask_path)} is {test_avg_dice}')