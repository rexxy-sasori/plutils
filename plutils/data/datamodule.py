from typing import Callable, Optional

import pl_bolts.datamodules as dms
import pytorch_lightning as pl
import torch
import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization
from torch.utils.data import DataLoader

from plutils.data.dataset import TarImageFolder
from plutils.data.utils import RGBToTensor, cifar100_normalization


def cifar10_datamodule(*args, **kwargs):
    return CIFARDataModule(cls=torchvision.datasets.CIFAR10, normalization=cifar10_normalization, *args, **kwargs)


def cifar100_datamodule(*args, **kwargs):
    return CIFARDataModule(cls=torchvision.datasets.CIFAR100, normalization=cifar100_normalization, *args, **kwargs)


def imgnet_datamodule(*args, **kwargs):
    dm = dms.ImagenetDataModule(*args, **kwargs)

    dm.prepare_data()
    dm.setup()

    dm._train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            RGBToTensor(),
            imagenet_normalization()
        ]
    )

    dm._val_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224 + 32),
            torchvision.transforms.CenterCrop(224),
            RGBToTensor(),
            imagenet_normalization()
        ]
    )

    dm._test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224 + 32),
            torchvision.transforms.CenterCrop(224),
            RGBToTensor(),
            imagenet_normalization()
        ]
    )

    return dm


def tar_imgnet_datamodule(*args, **kwargs):
    dm = TarImgNetDataModule(*args, **kwargs)
    dm.prepare_data()
    dm.setup()

    return dm


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, num_worker, batch_size, data_dir, drop_last, cls, normalization):
        super(CIFARDataModule, self).__init__()
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.root_dir = data_dir
        self.drop_last = drop_last
        self.cls = cls

        self.aug_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                RGBToTensor(),
                normalization(),
            ]
        )

        self.base_transform = torchvision.transforms.Compose(
            [
                RGBToTensor(),
                normalization(),
            ]
        )

    def prepare_data(self) -> None:
        self.cls(self.root_dir, train=True, download=True, transform=self.aug_transform)
        self.cls(self.root_dir, train=False, download=True, transform=self.base_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.cifar_train = self.cls(self.root_dir, train=True, download=True, transform=self.aug_transform)
        self.cifar_test = self.cls(self.root_dir, train=False, download=True, transform=self.base_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_worker)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_worker, drop_last=self.drop_last)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_worker, drop_last=self.drop_last)


class TarImgNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_tar_path,
                 val_tar_path,
                 num_imgs_per_val_class: int = 50,
                 image_size: int = 224,
                 num_workers: int = 0,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 *args, **kwargs):
        super(TarImgNetDataModule, self).__init__(*args, **kwargs)
        self.train_tar_path = train_tar_path
        self.val_tar_path = val_tar_path
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.image_size = image_size
        self.dims = (3, self.image_size, self.image_size)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_samples = 1281167 - self.num_imgs_per_val_class * self.num_classes

    @property
    def num_classes(self) -> int:
        return 1000

    def train_dataloader(self):
        transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        dataset = TarImageFolder(
            self.train_tar_path,
            split='train',
            transforms=transforms,
            num_val_images_per_class=self.num_imgs_per_val_class,
            num_classes=self.num_classes,
        )

        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def val_dataloader(self) -> DataLoader:
        transforms = self.val_transform() if self.val_transforms is None else self.val_transforms

        dataset = TarImageFolder(
            self.train_tar_path,
            split='val',
            transforms=transforms,
            num_val_images_per_class=self.num_imgs_per_val_class,
            num_classes=self.num_classes,
        )

        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        transforms = self.val_transform() if self.test_transforms is None else self.test_transforms

        dataset = TarImageFolder(
            self.val_tar_path,
            split='test',
            transforms=transforms,
            num_val_images_per_class=-1,
            num_classes=self.num_classes,
        )

        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

        return loader

    def train_transform(self) -> Callable:
        preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(self.image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                RGBToTensor(),
                imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:
        preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.image_size + 32),
                torchvision.transforms.CenterCrop(self.image_size),
                RGBToTensor(),
                imagenet_normalization(),
            ]
        )
        return preprocessing
