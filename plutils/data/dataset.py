import os
import re
import tarfile

import torch.utils.data.dataset
from PIL import Image

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _extract_tar_info(tarfile):
    class_to_idx = {}
    files = []
    targets = []
    for ti in tarfile.getmembers():
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        target = os.path.basename(dirname)
        class_to_idx[target] = None
        ext = os.path.splitext(basename)[1]
        if ext.lower() in IMG_EXTENSIONS:
            files.append(ti)
            targets.append(target)
    for idx, c in enumerate(sorted(class_to_idx.keys(), key=natural_key)):
        class_to_idx[c] = idx
    tarinfo_and_targets = zip(files, [class_to_idx[t] for t in targets])
    tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
    return tarinfo_and_targets


class TarImageFolder(torch.utils.data.dataset.Dataset):
    def __init__(
            self, root, split='train',
            num_val_images_per_class=50,
            num_classes=1000,
            transforms=None,
    ):
        assert os.path.isfile(root)
        self.root = root
        with tarfile.open(root) as tarf:  # cannot keep this open across processes, reopen later
            self.imgs = _extract_tar_info(tarf)

        if split in ['train', 'val']:
            train_imgs, val_imgs = self.partition_train_set(self.imgs, num_val_images_per_class)
            if split == 'train':
                self.imgs = train_imgs
            if split == 'val':
                self.imgs = val_imgs

        self.tarfile = None
        self.transforms = transforms
        self.num_classes = num_classes

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.imgs[index]
        iob = self.tarfile.extractfile(tarinfo)
        img = Image.open(iob).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def partition_train_set(self, imgs, nb_imgs_in_val):
        val = []
        train = []

        cts = {x: 0 for x in range(self.num_classes)}
        for img_name, idx in imgs:
            if cts[idx] < nb_imgs_in_val:
                val.append((img_name, idx))
                cts[idx] += 1
            else:
                train.append((img_name, idx))

        return train, val
