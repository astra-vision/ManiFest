import pytorch_lightning as pl
import torch
import os
from PIL import Image
from munch import Munch
from torchvision.transforms import ToPILImage, ToTensor
from networks import create_model

from argparse import ArgumentParser as AP

def main(ap):
        CHECKPOINT = ap.checkpoint
        OUTPUT_DIR = ap.output_dir
        INPUT_DIR = ap.input_dir
        # Load parameters
        #with open(os.path.join(root_dir, 'hparams.yaml')) as cfg_file:
        ckpt_path = torch.load(CHECKPOINT, map_location='cpu')
        hparams = ckpt_path['hyper_parameters']
        opt = Munch(hparams).opt
        print(opt.model)
        print(opt.seed)
        opt.phase = 'val'
        opt.no_flip = True
        # Load parameters to the model, load the checkpoint
        model = create_model(opt)
        model = model.load_from_checkpoint(CHECKPOINT)
        # Transfer the model to the GPU
        model.to('cuda')
        val_ds = INPUT_DIR
        image_list = os.listdir(val_ds)
        os.makedirs('{}/general'.format(OUTPUT_DIR), exist_ok=True)
        for index, im_path in enumerate(image_list):
            print('{}/{}:{}'.format(index + 1, len(image_list), im_path))
            original_image = Image.open(os.path.join(val_ds, im_path))
            original_size = original_image.size
            im = original_image.resize((480, 256), Image.BILINEAR)
            style_array = torch.randn(1, 8, 1, 1).cuda()
            im = ToTensor()(im) * 2 - 1
            im = im.cuda().unsqueeze(0)
            result = model.forward(im, style_array, type='global', ref_image=None)
            result = torch.clamp(result, -1, 1)
            img_global = ToPILImage()((result[0] + 1) / 2).resize(original_size, Image.BILINEAR)
            img_global.save('{}/general/{}'.format(OUTPUT_DIR, im_path))


if __name__ == '__main__':
        ap = AP()
        ap.add_argument('--checkpoint', required=True, type=str, help='checkpoint to load')
        ap.add_argument('--output_dir', required=True, type=str, help='where to save images')
        ap.add_argument('--input_dir', default='datasets/acdc_day2night/valRC', type=str, help='directory with images to translate')
        ap = ap.parse_args()
        main(ap)




