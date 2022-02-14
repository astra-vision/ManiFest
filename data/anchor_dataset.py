import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import pytorch_lightning as pl
import munch

def DatasetOptions():
    do = munch.Munch()
    do.serial_batches = False
    do.num_threads = 8
    do.batch_size = 1
    do.load_size = 480
    do.dataroot = 'dataset/day2twilight'
    do.crop_size = 0
    do.max_dataset_size = float('inf')
    do.preprocess = 'none'
    do.no_flip = False
    do.num_images = 25
    do.num_classes_A = 1
    do.num_classes_B = 2
    do.size_resize = (480, 256)

    return do

class AnchorDataset(BaseDataset):


    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A_id = os.path.join(opt.dataroot, opt.phase + 'S')  # create a path '/path/to/data/trainA'
        self.dir_A_m = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainB'
        self.dir_T = os.path.join(opt.dataroot, opt.phase + 'T')  # create a path '/path/to/data/trainB'

        self.S_paths = sorted(make_dataset(self.dir_A_id, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_m_paths = sorted(make_dataset(self.dir_A_m, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.T_paths = sorted(make_dataset(self.dir_T, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

        self.S_size = len(self.S_paths)  # get the size of dataset A
        self.A_m_size = len(self.A_m_paths)  # get the size of dataset B
        self.GT_size = len(self.T_paths)  # get the size of dataset B

        random.shuffle(self.S_paths)
        random.shuffle(self.A_m_paths)
        random.shuffle(self.T_paths)

        input_nc = self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.output_nc      # get the number of channels of output image
        self.transform_A_id = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_A_m = get_transform(self.opt, grayscale=(output_nc == 1))
        self.transform_T = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, S_paths and A_m_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            S_paths (str)    -- image paths
            A_m_paths (str)    -- image paths
        """
        S_path = self.S_paths[index % self.S_size]  # make sure index is within then range
        index_T = random.randint(0, self.opt.num_images - 1)
        class_anchor = random.randint(0, self.opt.num_classes_B - 1) # Multi-class support
        if class_anchor == 0:
            index_A = random.randint(0, len(self.A_m_paths) - 1)
            A_path = self.A_m_paths[index_A]
        elif class_anchor == 1:
            index_A = random.randint(0, len(self.S_paths) - 1)
            A_path = self.S_paths[index_A]

        T_path = self.T_paths[index_T]

        S_img = Image.open(S_path).convert('RGB').resize(self.opt.size_resize, Image.BILINEAR)
        A_img = Image.open(A_path).convert('RGB').resize(self.opt.size_resize, Image.BILINEAR)
        T_img = Image.open(T_path).convert('RGB').resize(self.opt.size_resize, Image.BILINEAR)

        S = self.transform_A_id(S_img)
        T = self.transform_T(T_img)

        A = self.transform_A_m(A_img)

        return {'S': S, 'A': A, 'class_anchor': class_anchor, 'T': T,
                'S_paths': S_path, 'A_m_paths': A_path, 'T_paths': T_path, 'S_class': 0}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.S_size, self.A_m_size)
