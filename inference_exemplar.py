import pytorch_lightning as pl
import torch
import os
from math import pi
from PIL import Image
from munch import Munch
from torchvision.transforms import ToPILImage, ToTensor
from networks import find_model_using_name, create_model

from argparse import ArgumentParser as AP

def main(ap):
        CHECKPOINT = ap.checkpoint
        OUTPUT_DIR = ap.output_dir
        INPUT_DIR = ap.input_dir
        REFERENCE_IMAGE = ap.reference_image
        # Load parameters
        #with open(os.path.join(root_dir, 'hparams.yaml')) as cfg_file:
        ckpt_path = torch.load(CHECKPOINT, map_location='cpu')
        hparams = ckpt_path['hyper_parameters']
        opt = Munch(hparams).opt
        opt.phase = 'val'
        opt.no_flip = True
        # Load parameters to the model, load the checkpoint
        model = create_model(opt)
        model = model.load_from_checkpoint(CHECKPOINT)
        # Transfer the model to the GPU
        model.to('cuda')
        val_ds = INPUT_DIR

        im_ref = Image.open(REFERENCE_IMAGE).resize((480, 256), Image.BILINEAR)
        im_ref = ToTensor()(im_ref) * 2 - 1
        im_ref = im_ref.cuda().unsqueeze(0)

        os.makedirs('{}/exemplar'.format(OUTPUT_DIR), exist_ok=True)
        for index, im_path in enumerate(os.listdir(val_ds)):
            print(index)
            im = Image.open(os.path.join(val_ds, im_path)).resize((480, 256), Image.BILINEAR)
            im = ToTensor()(im) * 2 - 1
            im = im.cuda().unsqueeze(0)
            style_array = torch.randn(1, 8, 1, 1).cuda()
            result = model.forward(im, style_array, type='exemplar', ref_image=im_ref)
            result = torch.clamp(result, -1, 1)
            img_global = ToPILImage()((result[0] + 1) / 2)
            img_global.save('{}/exemplar/{}'.format(OUTPUT_DIR, im_path))

if __name__ == '__main__':
        ap = AP()
        ap.add_argument('--checkpoint', required=True, type=str, help='checkpoint to load')
        ap.add_argument('--output_dir', required=True, type=str, help='where to save images')
        ap.add_argument('--input_dir', default='datasets/acdc_day2night/valRC', type=str, help='directory with images to translate')
        ap.add_argument('--reference_image', required=True, type=str, help='reference_image')
        ap = ap.parse_args()
        main(ap)




