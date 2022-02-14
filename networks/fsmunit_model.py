'''
In the provided file,
Source images are identified by "A"
Anchor images are identified by "B"
Few-shot target by "GT"

We exploit the additional loss described in the supp, called loss_vgg_gan
L_style is called loss_vgg_style
'''

import torch
import itertools
from .base_model import BaseModel
from .backbones import fsmunit as networks
import munch
import random
import kornia.augmentation as K

def ModelOptions():
    mo = munch.Munch()
    # Generator
    mo.gen_dim = 64
    mo.style_dim = 8
    mo.gen_activ = 'relu'
    mo.n_downsample = 2
    mo.n_res = 4
    mo.gen_pad_type = 'reflect'
    mo.mlp_dim = 256

    # Discriminiator
    mo.disc_dim = 64
    mo.disc_norm = 'none'
    mo.disc_activ = 'lrelu'
    mo.disc_n_layer = 4
    mo.num_scales = 3
    mo.disc_pad_type = 'reflect'

    # Initialization
    mo.init_type_gen = 'kaiming'
    mo.init_type_disc = 'normal'
    mo.init_gain = 0.02

    # Weights
    mo.lambda_gan = 1
    mo.lambda_gan_patches = 1
    mo.lambda_rec_image = 10
    mo.lambda_rec_style = 1
    mo.lambda_rec_content = 1
    mo.lambda_rec_cycle = 10
    mo.lambda_vgg = 0.5

    mo.lambda_vgg_style = 2e-4
    mo.lambda_vgg_fs_res = 0.5
    mo.lambda_vgg_fs = 0.5
    return mo

class FSMunitModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'rec_A', 'rec_style_A', 'rec_content_A', 'vgg_A','G_patches', 'D_patches',
                           'D_B', 'G_B', 'cycle_B', 'rec_B', 'rec_style_B', 'rec_content_B', 'vgg_B', 'vgg_style']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'fake_B_weights', 'fake_B_residual', 'fake_B_style', 'rec_A_img', 'rec_A_cycle']
        visual_names_B = ['real_B', 'fake_A', 'rec_B_img', 'rec_B_cycle', 'real_GT']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']


        self.netG_A = networks.define_G_munit(opt.input_nc, opt.output_nc, opt.gen_dim, opt.style_dim, opt.n_downsample,
                                              opt.n_res, opt.gen_pad_type, opt.mlp_dim, opt.gen_activ, opt.init_type_gen,
                                              opt.init_gain, self.gpu_ids, num_classes=self.opt.num_classes_A)
        self.netG_B = networks.define_G_munit(opt.input_nc, opt.output_nc, opt.gen_dim, opt.style_dim, opt.n_downsample,
                                              opt.n_res, opt.gen_pad_type, opt.mlp_dim, opt.gen_activ, opt.init_type_gen,
                                              opt.init_gain, self.gpu_ids, num_classes=self.opt.num_classes_B)

        self.netD_A = networks.define_D_munit(opt.output_nc, opt.disc_dim, opt.disc_norm, opt.disc_activ, opt.disc_n_layer,
                                              opt.gan_mode, opt.num_scales, opt.disc_pad_type, opt.init_type_disc,
                                              opt.init_gain, self.gpu_ids, num_classes=self.opt.num_classes_B)
        self.netD_B = networks.define_D_munit(opt.output_nc, opt.disc_dim, opt.disc_norm, opt.disc_activ, opt.disc_n_layer,
                                              opt.gan_mode, opt.num_scales, opt.disc_pad_type, opt.init_type_disc,
                                              opt.init_gain, self.gpu_ids, num_classes=self.opt.num_classes_A)

        # Patch-based discriminator
        self.netD_patches = networks.define_D_munit(opt.output_nc, opt.disc_dim, opt.disc_norm, opt.disc_activ, opt.disc_n_layer,
                                              opt.gan_mode, 2, opt.disc_pad_type, opt.init_type_disc,
                                              opt.init_gain, self.gpu_ids, num_classes=1)

        self.num_classes_B = self.opt.num_classes_B
        self.num_classes_A = self.opt.num_classes_A

        if opt.lambda_vgg > 0:
            self.instance_norm = torch.nn.InstanceNorm2d(512)
            self.vgg = networks.Vgg16()
            self.vgg.load_state_dict(torch.load('res/vgg_imagenet.pth'))
            self.vgg.to(self.device)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        self.weights = torch.nn.Parameter(torch.zeros(self.num_classes_B).cuda())

        self.augmentations_patches = torch.nn.Sequential(
            K.RandomHorizontalFlip(),
            K.RandomAffine(360, p=1),
            K.RandomCrop(size=(64, 64))
        )

        self.alternate = False

    def configure_optimizers(self):
        opt_G = torch.optim.Adam([{'params': self.netG_A.parameters()}, {'params': self.netG_B.parameters()}, {'params': [self.weights], 'lr': 0.01}],
                                 weight_decay=0.0001, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        opt_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_patches.parameters()),
                                            weight_decay=0.0001, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        scheduler_G = self.get_scheduler(self.opt, opt_G)
        scheduler_D = self.get_scheduler(self.opt, opt_D)
        return [opt_D, opt_G], [scheduler_D, scheduler_G]

    def reconCriterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = input['S']
        self.real_B = input['A']
        self.class_B = input['class_anchor']
        self.class_A = input['S_class']
        self.real_GT = input['T']
        self.image_paths = input['S_paths']
        self.random_class_B = random.randint(0, self.num_classes_B - 1)
        self.random_class_A = random.randint(0, self.num_classes_A - 1)

    def __vgg_preprocess(self, batch):
        tensortype = type(batch)
        (r, g, b) = torch.chunk(batch, 3, dim=1)
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
        batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
        mean = tensortype(batch.data.size()).to(self.device)
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(mean)  # subtract mean
        return batch

    def __calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def __compute_vgg_loss_onlyperceptual(self, img, target):
        img_vgg = self.__vgg_preprocess(img)
        target_vgg = self.__vgg_preprocess(target)
        img_fea = self.vgg(img_vgg)
        target_fea = self.vgg(target_vgg)
        return torch.mean((self.instance_norm(img_fea) - self.instance_norm(target_fea)) ** 2)

    def __get_vgg_style_loss(self, f1, f2):
        style_loss = 0
        for x, y in zip(f1, f2):
            x_mean, x_std = self.__calc_mean_std(x)
            y_mean, y_std = self.__calc_mean_std(y)
            style_loss += torch.mean((x_mean - y_mean) ** 2) + torch.mean((x_std - y_std) ** 2)
        return style_loss

    def get_fewshot_style_code(self, im):
        _, f = self.vgg(im, with_style=True)
        means, stds = [],[]
        for x in f:
            x_mean, x_std = self.__calc_mean_std(x)
            means.append(x_mean.squeeze())
            stds.append(x_std.squeeze())
        style_feat = torch.cat(means + stds, 0)
        return style_feat.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def __compute_vgg_loss(self, img, target, style):
        img_vgg = self.__vgg_preprocess(img)
        target_vgg = self.__vgg_preprocess(target)
        style_vgg = self.__vgg_preprocess(style)
        img_fea, style_feat_img = self.vgg(img_vgg, with_style=True)
        target_fea = self.vgg(target_vgg)
        _, style_feat_style = self.vgg(style_vgg, with_style=True)
        content_loss = torch.mean((self.instance_norm(img_fea) - self.instance_norm(target_fea)) ** 2)
        style_loss = self.__get_vgg_style_loss(style_feat_img, style_feat_style)
        return content_loss + style_loss * self.opt.lambda_vgg_style

    def forward(self, im, style_B_fake = None, type='global', ref_image=None, weights = None):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Random style sampling
        if style_B_fake is None:
            style_B_fake = torch.randn(im.size(0), self.opt.style_dim, 1, 1).to(self.device)

        if weights is None:
            weights = torch.softmax(self.weights, 0)

        # Encoding
        self.content_A, self.style_A_real = self.netG_A.encode(im)
        if type == 'global':
            self.fake_B_weights, self.fake_B_residual = self.netG_B.decode_weighted_global(self.content_A,
                                                                                    style_B_fake,
                                                                                    style_B_fake,
                                                                                    weights)
        elif type == 'exemplar':

            self.fake_B_weights, self.fake_B_residual = self.netG_B.decode_weighted_exemplar(self.content_A,
                                                                                             style_B_fake,
                                                                                             self.get_fewshot_style_code(
                                                                                                 ref_image),
                                                                                             weights)
        else:
            raise NotImplementedError

        self.fake_B = self.fake_B_weights + self.fake_B_residual
        return self.fake_B

    def forward_train(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # Random style sampling
        self.style_A_fake = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
        self.style_B_fake = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)

        # Encoding
        self.content_A, self.style_A_real = self.netG_A.encode(self.real_A)
        self.content_B, self.style_B_real = self.netG_B.encode(self.real_B)

        # Reconstruction
        self.rec_A_img = self.netG_A.decode(self.content_A, self.style_A_real, self.class_A)
        self.rec_B_img = self.netG_B.decode(self.content_B, self.style_B_real, self.class_B)

        # Cross domain
        self.fake_B = self.netG_B.decode(self.content_A, self.style_B_fake, self.class_B)
        self.fake_A = self.netG_A.decode(self.content_B, self.style_A_fake, self.class_A)

        # Re-encoding everyting
        self.rec_content_B, self.rec_style_A = self.netG_A.encode(self.fake_A)
        self.rec_content_A, self.rec_style_B = self.netG_B.encode(self.fake_B)

        if self.opt.lambda_rec_cycle > 0:
            self.rec_A_cycle = self.netG_A.decode(self.rec_content_A, self.style_A_real, self.class_A)
            self.rec_B_cycle = self.netG_B.decode(self.rec_content_B, self.style_B_real, self.class_B)
        if not self.alternate:
            self.fake_B_weights, self.fake_B_residual = self.netG_B.decode_weighted_exemplar(self.content_A,
                                                                                    self.style_B_fake,
                                                                                    self.get_fewshot_style_code(self.real_GT),
                                                                                    torch.softmax(self.weights, 0))
            self.alternate = True
        else:
            self.fake_B_weights, self.fake_B_residual = self.netG_B.decode_weighted_global(self.content_A,
                                                                                    self.style_B_fake,
                                                                                    self.style_B_fake,
                                                                                    torch.softmax(self.weights, 0))
            self.alternate = False
        self.fake_B_style = self.fake_B_weights + self.fake_B_residual
        self.fake_B_style = torch.clamp(self.fake_B_style, -1, 1)

    def training_step_D(self):
        with torch.no_grad():
            # Random style sampling
            self.style_A_fake = torch.randn(self.real_A.size(0), self.opt.style_dim, 1, 1).to(self.device)
            self.style_B_fake = torch.randn(self.real_B.size(0), self.opt.style_dim, 1, 1).to(self.device)

            # Encoding
            self.content_A, self.style_A_real = self.netG_A.encode(self.real_A)
            self.content_B, self.style_B_real = self.netG_B.encode(self.real_B)

            self.fake_B = self.netG_B.decode(self.content_A, self.style_B_fake, self.class_B)
            self.fake_A = self.netG_A.decode(self.content_B, self.style_A_fake, self.class_A)
            if not self.alternate:
                self.fake_B_weights, self.fake_B_residual = self.netG_B.decode_weighted_exemplar(self.content_A,self.style_B_fake, self.get_fewshot_style_code(self.real_GT), torch.softmax(self.weights, 0))
            else:
                self.fake_B_weights, self.fake_B_residual = self.netG_B.decode_weighted_global(self.content_A,self.style_B_fake, self.style_B_fake, torch.softmax(self.weights, 0))
            self.fake_B_style = self.fake_B_weights + self.fake_B_residual

        self.loss_D_A = self.netD_A.calc_dis_loss(self.fake_B, self.real_B, self.class_B, self.device) * self.opt.lambda_gan
        self.loss_D_B = self.netD_B.calc_dis_loss(self.fake_A, self.real_A, self.class_A, self.device) * self.opt.lambda_gan
        self.loss_D_patches = self.netD_patches.calc_dis_loss(self.augmentations_patches(self.fake_B_style),
                                                              self.augmentations_patches(self.real_GT),
                                                              0, self.device)* self.opt.lambda_gan_patches
        loss_D = self.loss_D_A + self.loss_D_B + self.loss_D_patches
        return loss_D


    def training_step_G(self):
        self.forward_train()
        self.loss_rec_A = self.reconCriterion(self.rec_A_img, self.real_A) * self.opt.lambda_rec_image
        self.loss_rec_B = self.reconCriterion(self.rec_B_img, self.real_B) * self.opt.lambda_rec_image

        self.loss_rec_style_A = self.reconCriterion(self.rec_style_A, self.style_A_fake) * self.opt.lambda_rec_style
        self.loss_rec_style_B = self.reconCriterion(self.rec_style_B, self.style_B_fake) * self.opt.lambda_rec_style

        self.loss_rec_content_A = self.reconCriterion(self.rec_content_A, self.content_A) * self.opt.lambda_rec_content
        self.loss_rec_content_B = self.reconCriterion(self.rec_content_B, self.content_B) * self.opt.lambda_rec_content

        if self.opt.lambda_rec_cycle > 0:
            self.loss_cycle_A = self.reconCriterion(self.rec_A_cycle, self.real_A) * self.opt.lambda_rec_cycle
            self.loss_cycle_B = self.reconCriterion(self.rec_B_cycle, self.real_B) * self.opt.lambda_rec_cycle
        else:
            self.loss_cycle_A = 0
            self.loss_cycle_B = 0

        self.loss_G_A = self.netD_A.calc_gen_loss(self.fake_B, self.class_B, self.device) * self.opt.lambda_gan
        self.loss_G_B = self.netD_B.calc_gen_loss(self.fake_A, self.class_A, self.device) * self.opt.lambda_gan
        self.loss_G_patches = self.netD_patches.calc_gen_loss(
            self.augmentations_patches(self.fake_B_style), 0, self.device) * self.opt.lambda_gan_patches

        if self.opt.lambda_vgg > 0:
            self.loss_vgg_A = self.__compute_vgg_loss_onlyperceptual(self.fake_A, self.real_B) * self.opt.lambda_vgg
            self.loss_vgg_B = self.__compute_vgg_loss_onlyperceptual(self.fake_B, self.real_A) * self.opt.lambda_vgg
        else:
            self.loss_vgg_A = 0
            self.loss_vgg_B = 0

        self.loss_vgg_style = self.__compute_vgg_loss(self.fake_B_style, self.real_A, self.real_GT) * self.opt.lambda_vgg_fs_res
        self.loss_vgg_gan = self.__compute_vgg_loss(self.fake_B_weights, self.real_A, self.real_GT) * self.opt.lambda_vgg_fs # Using the manifold to get nearer

        self.loss_G = self.loss_rec_A + self.loss_rec_B + self.loss_rec_style_A + self.loss_rec_style_B + \
            self.loss_rec_content_A + self.loss_rec_content_B + self.loss_cycle_A + self.loss_cycle_B + \
            self.loss_G_A + self.loss_G_B + self.loss_vgg_A + self.loss_vgg_B + self.loss_vgg_style + \
                      self.loss_G_patches + self.loss_vgg_gan

        return self.loss_G

    def training_step(self, batch, batch_idx, optimizer_idx):

        self.set_input(batch)
        if optimizer_idx == 0:
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_patches], True)
            self.set_requires_grad([self.netG_A, self.netG_B], False)

            return self.training_step_D()
        elif optimizer_idx == 1:
            self.set_requires_grad([self.netD_A, self.netD_B, self.netD_patches], False)  # Ds require no gradients when optimizing Gs
            self.set_requires_grad([self.netG_A, self.netG_B], True)

            return self.training_step_G()
