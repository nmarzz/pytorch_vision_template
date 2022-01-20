''' Load datasets into Pytorch DataLoader objects.

Includes dataset specific loaders and a getter function to access them
'''

import os
import yaml
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)
imagenet_path = parsed_config['imagenet_path']
tiny_imagenet_path = parsed_config['tiny_imagenet_path']


DATA_ROOT = parsed_config['data_dir']


def imagenet_loader(batch_size: int, distributed: bool = False) -> tuple([DataLoader, DataLoader]):
    """Loader for the ImageNet dataset.

    Args:
        batch_size: Batch size.
        workers: Number of workers.
        distributed: Whether or not to parallelize.

    Returns:
        Training and validation set loaders.

    """
    traindir = os.path.join(imagenet_path, 'train')
    valdir = os.path.join(imagenet_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_set = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=10,
        pin_memory=True, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=10)

    return train_loader, val_loader


def tiny_imagenet_loader(batch_size: int, distributed: bool = False) -> tuple([DataLoader, DataLoader]):
    """Loader for the Tiny ImageNet dataset.

    Args:
        batch_size: Batch size.
        workers: Number of workers.
        distributed: Whether or not to parallelize.

    Returns:
        Training and validation set loaders.

    """
    traindir = os.path.join(tiny_imagenet_path, 'train')
    valdir = os.path.join(tiny_imagenet_path, 'val')

    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])

    train_set = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=10,
        pin_memory=True, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=10)

    return train_loader, val_loader


def mnist_loader(batch_size: int, distributed: bool) -> None:
    train_set = datasets.MNIST(DATA_ROOT, train=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]), download=True)
    val_set = datasets.MNIST(DATA_ROOT, train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]), download=True)

    # For distributed training
    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=8,
        pin_memory=True, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=8)

    return train_loader, val_loader


def fashion_mnist_loader(batch_size: int, distributed: bool) -> None:
    train_set = datasets.FashionMNIST(DATA_ROOT, train=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]), download=True)
    val_set = datasets.FashionMNIST(DATA_ROOT, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                    ]), download=True)

    # For distributed training
    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=8,
        pin_memory=True, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=8)

    return train_loader, val_loader


# def cifar10_loader(batch_size: int, distributed: bool, resize: bool) -> tuple([DataLoader, DataLoader]):
#     # Get the train/val sets

#     if resize:
#         train_transforms = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#         val_transforms = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#     else:
#         train_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#         val_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])

#     train_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=True,
#                                              transform=train_transforms, download=True)

#     val_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=False,
#                                            transform=val_transforms, download=True)

#     # For distributed training
#     if distributed:
#         sampler = DistributedSampler(train_set)
#     else:
#         sampler = None

#     # Now make the loaders
#     train_loader = DataLoader(
#         train_set, batch_size=batch_size, sampler=sampler, num_workers=8,
#         pin_memory=True, shuffle=(not distributed))

#     val_loader = DataLoader(val_set, batch_size=batch_size,
#                             shuffle=False, num_workers=8)

#     return train_loader, val_loader


import ffcv.transforms as ftrans
import ffcv.fields.decoders as decode
import ffcv.loader as floader
import ffcv.transforms.common as ftranscommon

def cifar10_loader(batch_size: int, distributed: bool, resize: bool, device) -> tuple([DataLoader, DataLoader]):
    # Get the train/val sets

    # Note that statistics are wrt to uin8 range, [0,255].
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    

    loaders = {}
    for name in ['train', 'val']:
        label_pipeline = [decode.IntDecoder(), ftrans.ToTensor(), ftrans.ToDevice(device), ftranscommon.Squeeze()]
        image_pipeline = [decode.SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == 'train':
            image_pipeline.extend([
                ftrans.RandomHorizontalFlip(),
                ftrans.RandomTranslate(padding=2),
                ftrans.Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ftrans.ToTensor(),
            ftrans.ToDevice(device, non_blocking=True),
            ftrans.ToTorchImage(),
            ftrans.Convert(torch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Create loaders
        loaders[name] = floader.Loader(f'data/cifar10/{name}_set/ds.beton',
                                batch_size=batch_size,
                                num_workers=8,
                                order=floader.OrderOption.RANDOM,
                                drop_last=(name == 'train'),
                                distributed = distributed,
                                pipelines={'image': image_pipeline,
                                        'label': label_pipeline})
    return loaders['train'], loaders['val']


def cifar100_loader(batch_size: int, distributed: bool, resize: bool) -> tuple([DataLoader, DataLoader]):
    # Get the train/val sets

    if resize:
        train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    train_set = torchvision.datasets.CIFAR100(DATA_ROOT, train=True,
                                              transform=train_transforms, download=True)

    val_set = torchvision.datasets.CIFAR100(DATA_ROOT, train=False,
                                            transform=val_transforms, download=True)

    # For distributed training
    if distributed:
        sampler = DistributedSampler(train_set)
    else:
        sampler = None

    # Now make the loaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler, num_workers=8,
        pin_memory=True, shuffle=(not distributed))

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=8)

    return train_loader, val_loader


def get_loader(name: str, batch_size: int, distributed: bool, resize: bool, device):
    if name == 'fashion_mnist':
        return fashion_mnist_loader(batch_size=batch_size, distributed=distributed)
    elif name == 'mnist':
        return mnist_loader(batch_size=batch_size, distributed=distributed)
    elif name == 'cifar10':
        return cifar10_loader(batch_size=batch_size, distributed=distributed, resize=resize,device = device)
    elif name == 'cifar100':
        return cifar100_loader(batch_size=batch_size, distributed=distributed, resize=resize)
    elif name == 'imagenet':
        return imagenet_loader(batch_size=batch_size, distributed=distributed)
    elif name == 'tiny_imagenet':
        return tiny_imagenet_loader(batch_size=batch_size, distributed=distributed)
    else:
        raise ValueError(f'{name} dataset is not available')
