from PIL import Image, ImageFile
import cv2
import os
import random
import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import DepthNorm
from torch.utils.data.dataloader import default_collate

def my_collate_fn(batch):
    #  fileter NoneType data
    batch = list(filter(lambda x:x['depth'] is not None and x['image'] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)

class DepthDataLoader(object):
    def __init__(self, args, mode, on_iridis5=False):
        # transforms.ToTensor()
        transform_NYU = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.53607797, 0.53617338, 0.53618207], std=[0.31895092, 0.31896688, 0.31896867])
        ]
        )

        if mode == 'train':
            # self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.training_samples = DataLoadPreprocess(args, mode, transform=transform_NYU, on_iridis5=on_iridis5)
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'target':
            # self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.testing_samples = DataLoadPreprocess(args, mode, transform=transform_NYU, on_iridis5=on_iridis5)
            if args.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, args.batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=transform_NYU, on_iridis5=on_iridis5)
            if args.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler,
                                   collate_fn=my_collate_fn)



class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False, on_iridis5=False):
        self.args = args
        if mode == 'target' or mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        # self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.on_iridis5 = on_iridis5

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        if self.mode == 'train':
            if self.on_iridis5:
                image_path = sample_path.split()[0].replace(sample_path.split()[0].split('/3D60/dataset')[0], '/scratch/yw10y19/Datasets')
                depth_path = sample_path.split()[1].replace(sample_path.split()[1].split('/3D60/dataset')[0], '/scratch/yw10y19/Datasets')
            else:
                image_path = os.path.join(self.args.data_path, sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, sample_path.split()[1])

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = image.transpose(2, 0, 1)
            else:
                image = None

            # for exr file, 3 channels are the same
            if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval, doNorm=False)
            else:
                depth_gt = None

            sample = {'image': image, 'depth': depth_gt}

        if self.mode == 'target':
            image_path = os.path.join(self.args.data_path, sample_path.split()[0])
            depth_path = os.path.join(self.args.data_path, sample_path.split()[1])

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = image.transpose(2, 0, 1)
            else:
                image = None

            # for exr file, 3 channels are the same
            if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval, doNorm=False)
            else:
                depth_gt = None

            sample = {'image': image, 'depth': depth_gt}

        if self.mode == 'online_eval':
            image_path = self.args.data_path + sample_path.split()[0]
            depth_path = self.args.data_path + sample_path.split()[1]

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is not None:
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    image = image.transpose(2, 0, 1)
            else:
                image = None

            # for exr file, 3 channels are the same
            if cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) is not None:
                depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:, :, 0:1]
                depth_gt = DepthNorm(depth_gt, minDepth=self.args.min_depth_eval, maxDepth=self.args.max_depth_eval,
                                     doNorm=False)
            else:
                depth_gt = None

            sample = {'image': image, 'depth': depth_gt}

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


