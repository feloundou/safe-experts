import os
import torch
# from datasets import get_dataset, get_dataloader
# from dnn.models.ALAE import MLP_ALAE
from utils.common_utils import get_config_str
import argparse
from pprint import pprint
from utils.latent_interpolation import plot_latent_interpolation


import numpy as np
import torch
import torch.tensor
import torch.utils
import torch.utils.data
from torchvision.datasets.utils import download_and_extract_archive, download_file_from_google_drive
from torchvision.datasets.mnist import read_image_file
from torch.utils.data import Dataset
from torch.utils.data import random_split
import os
import tarfile
import cv2
from tqdm import tqdm

import os
from tqdm import tqdm
from torchvision.utils import save_image
from dnn.sub_modules.AlaeModules import *
# from dnn.sub_modules.StyleGanGenerator import StylleGanGenerator, MappingFromLatent
from utils.tracker import LossTracker
from dnn.costume_layers import compute_r1_gradient_penalty
from datasets import get_dataloader, EndlessDataloader




MNIST_WORKING_DIM=28
VAL_SET_PORTION=0.05





COMMON_DEFAULT = {"g_penalty_coeff": 10,
                  'descriminator_layers': 3,
                  'mapping_lr_factor': 0.01, # StyleGan paper: "... reduce the learning rate by two orders of magnitude for the mapping network..."
                  'discriminator_lr_factor':0.1 # Found in The ALAE official implemenation.
                  }



class DiscriminatorMLP(nn.Module):
    """
    An n MLP layers discriminator  with leaky Relu. See
    """
    def __init__(self, num_layers, input_dim=256):
        super(DiscriminatorMLP, self).__init__()
        assert num_layers >= 2
        layers = []
        for i in range(num_layers):
            out_dim = 1 if i == num_layers - 1 else input_dim
            layers += [LREQ_FC_Layer(input_dim, out_dim), nn.LeakyReLU(0.2)]
        self.mapping = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        x = x.view(-1)
        return x


class EncoderMLP(nn.Module):
    """
    The MLP version of ALAE encoder
    """
    def __init__(self, input_img_dim, latent_dim):
        super(EncoderMLP, self).__init__()
        self.out_dim = latent_dim
        self.input_img_dim = input_img_dim

        self.fc_1 = LREQ_FC_Layer(input_img_dim ** 2, 1024)
        self.fc_2 = LREQ_FC_Layer(1024, 1024)
        self.fc_3 = LREQ_FC_Layer(1024, latent_dim)

    def encode(self, x):
        x = x.view(x.shape[0], self.input_img_dim**2)

        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)
        x = F.leaky_relu(x, 0.2)

        return x

    def forward(self, x):
        return self.encode(x)


class GeneratorMLP(nn.Module):
    """
    MLP version of ALAE generator.
    """
    def __init__(self, latent_dim, output_img_dim):
        super(GeneratorMLP, self).__init__()
        self.latent_size = latent_dim
        self.output_img_dim = output_img_dim

        self.fc_1 = LREQ_FC_Layer(latent_dim, 1024)
        self.fc_2 = LREQ_FC_Layer(1024, 1024)
        self.fc_3 = LREQ_FC_Layer(1024, self.output_img_dim ** 2)

    def forward(self, x):
        x = self.fc_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.fc_3(x)

        x = x.view(x.shape[0], 1, self.output_img_dim, self.output_img_dim)

        return x


class AlaeStyleEncoderBlock(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, downsample=False, is_last_block=False):
        super(AlaeStyleEncoderBlock, self).__init__()
        assert not (is_last_block and downsample), "You should not downscale after last block"
        self.downsample = downsample
        self.is_last_block = is_last_block
        self.conv1 = Lreq_Conv2d(in_channels, in_channels, 3, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.instance_norm_1 = StyleInstanceNorm2d(in_channels)
        self.c_1 = LREQ_FC_Layer(2 * in_channels, latent_dim)
        if is_last_block:
            self.conv2 = Lreq_Conv2d(in_channels, out_channels, STARTING_DIM, 0)
            self.c_2 = LREQ_FC_Layer(out_channels, latent_dim)
        else:
            scale = 2 if downsample else 1
            self.conv2 = torch.nn.Sequential(LearnablePreScaleBlur(in_channels),
                                             Lreq_Conv2d(in_channels, out_channels, 3, 1),
                                             torch.nn.AvgPool2d(scale, scale))
            self.instance_norm_2 = StyleInstanceNorm2d(out_channels)
            self.c_2 = LREQ_FC_Layer(2 * out_channels, latent_dim)

        self.name = f"EncodeBlock({latent_dim}, {in_channels}, {out_channels}, is_last_block={is_last_block}, downsample={downsample})"

    def __str__(self):
        return self.name

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)

        x, style_1 = self.instance_norm_1(x)
        w1 = self.c_1(style_1.squeeze(3).squeeze(2))

        x = self.conv2(x)
        x = self.lrelu(x)
        if self.is_last_block:
            w2 = self.c_2(x.squeeze(3).squeeze(2))
        else:
            x, style_2 = self.instance_norm_2(x)
            w2 = self.c_2(style_2.squeeze(3).squeeze(2))

        return x, w1, w2


class AlaeEncoder(nn.Module):
    """
    The Style version of ALAE encoder
    """
    def __init__(self, latent_dim, progression):
        """
        progression: A list of tuples (<out_res>, <out_channels>) that describes the Encoding blocks this module should have
        """
        super().__init__()
        assert progression[0][0] == STARTING_DIM, f"Last module should note downscale so first out_dim should be {STARTING_DIM}"
        self.latent_size = latent_dim
        self.from_rgbs = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        # Parse the module description given in "progression"
        for i in range(len(progression)):
            self.from_rgbs.append(Lreq_Conv2d(3, progression[i][1], 1, 0))
        for i in range(len(progression) - 1, -1, -1):
            if i == 0:
                self.conv_blocks.append(AlaeStyleEncoderBlock(latent_dim, progression[i][1], STARTING_CHANNELS,
                                                              is_last_block=True))
            else:
                downsample = progression[i][0] / 2 == progression[i - 1][0]
                self.conv_blocks.append(AlaeStyleEncoderBlock(latent_dim, progression[i][1], progression[i - 1][1],
                                                              downsample=downsample))


        assert(len(self.conv_blocks) == len(self.from_rgbs))
        self.n_layers = len(self.conv_blocks)

    def __str__(self):
        name = "Style-Encoder:\n"
        name += "\tfromRgbs\n"
        for i in range(len(self.from_rgbs)):
            name += f"\t {self.from_rgbs[i]}\n"
        name += "\tStyleEncoderBlocks\n"
        for i in range(len(self.conv_blocks)):
            name += f"\t {self.conv_blocks[i]}\n"
        return name

    def forward(self, image, final_resolution_idx, alpha=1):
        latent_vector = torch.zeros(image.shape[0], self.latent_size).to(image.device)

        feature_maps = self.from_rgbs[final_resolution_idx](image)

        first_layer_idx = self.n_layers - final_resolution_idx - 1
        for i in range(first_layer_idx, self.n_layers):
            feature_maps, w1, w2 = self.conv_blocks[i](feature_maps)
            latent_vector += w1 + w2

            # If this is the first conv block to be run and this is not the last one the there is an already stabilized
            # previous scale layers : Alpha blend the output of the unstable new layer with the downscaled putput
            # of the previous one
            if i == first_layer_idx and i != self.n_layers - 1 and alpha < 1:
                if self.conv_blocks[i].downsample:
                    image = downscale_2d(image)
                skip_first_block_feature_maps = self.from_rgbs[final_resolution_idx - 1](image)
                feature_maps = alpha * feature_maps + (1 - alpha) * skip_first_block_feature_maps

        return latent_vector


class ALAE:
    """
    A generic (ASTRACT) Implementation of https://arxiv.org/abs/2004.04467
    """
    def __init__(self, model_config, architecture_mode, device):
        self.device = device
        self.cfg = COMMON_DEFAULT
        self.cfg.update(model_config)

        if architecture_mode == "MLP":

            self.G = GeneratorMLP(latent_dim=self.cfg['w_dim'], output_img_dim=self.cfg['image_dim']).to(device).train()
            self.E = EncoderMLP(input_img_dim=self.cfg['image_dim'], latent_dim=self.cfg['w_dim']).to(device).train()
        else:
            progression = list(zip(self.cfg['resolutions'], self.cfg['channels']))
            self.G = StylleGanGenerator(latent_dim=self.cfg['w_dim'], progression=progression).to(device).train()
            self.E = AlaeEncoder(latent_dim=self.cfg['w_dim'], progression=progression).to(device).train()

        self.F = MappingFromLatent(input_dim=self.cfg['z_dim'], out_dim=self.cfg['w_dim'], num_layers=self.cfg['mapping_layers']).to(device).train()
        self.D = DiscriminatorMLP(input_dim=self.cfg['w_dim'], num_layers=self.cfg['descriminator_layers']).to(device).train()

        self.ED_optimizer = torch.optim.Adam([{'params': self.D.parameters(), 'lr_mult': self.cfg['discriminator_lr_factor']},
                                      {'params': self.E.parameters(),}],
                                      betas=(0.0, 0.99), weight_decay=0)
        self.FG_optimizer = torch.optim.Adam([{'params': self.F.parameters(), 'lr_mult': self.cfg['mapping_lr_factor']},
                                      {'params': self.G.parameters()}],
                                      betas=(0.0, 0.99), weight_decay=0)

    def __str__(self):
        return f"F\n{self.F}\nG\n{self.G}\nE\n{self.E}\nD\n{self.D}\n"

    def set_optimizers_lr(self, new_lr):
        """
        resets the learning rate of the optimizers.
        lr_mult allows rescaling specifoic param groups.
        The StyleGan paper describes the lr scale of theMapping layers:
        "We thus reduce the learning rate by two orders of magnitude for the mapping network, i.e., λ = 0.01 ·λ"
        The decrease of the Discriminator D is just a parameter found in the official implementation.
        """
        for optimizer in [self.ED_optimizer, self.FG_optimizer]:
            for group in optimizer.param_groups:
                mult = group.get('lr_mult', 1)
                group['lr'] = new_lr * mult

    def get_ED_loss(self, batch_real_data, **ae_kwargs):
        """
        Computes a standard adverserial loss for the dictriminator D(E( * )):
          how much  D(E( * )) can differentiate between real images and images generated by G(F( * ))
         """
        batch_z = torch.randn(batch_real_data.shape[0], self.cfg['z_dim'], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            batch_fake_data = self.G(self.F(batch_z), **ae_kwargs)
        fake_images_dicriminator_outputs = self.D(self.E(batch_fake_data, **ae_kwargs))
        real_images_dicriminator_outputs = self.D(self.E(batch_real_data, **ae_kwargs))
        loss = F.softplus(fake_images_dicriminator_outputs) + F.softplus(-real_images_dicriminator_outputs)

        r1_penalty = compute_r1_gradient_penalty(real_images_dicriminator_outputs, batch_real_data)

        loss += self.cfg['g_penalty_coeff'] * r1_penalty
        loss = loss.mean()

        return loss

    def get_FG_loss(self, batch_real_data, **ae_kwargs):
        """
        Computes a standard adverserial loss for the generator:
            how much  G(F( * )) can fool D(E ( * ))
        """
        batch_z = torch.randn(batch_real_data.shape[0], self.cfg['z_dim'], dtype=torch.float32).to(self.device)
        batch_fake_data = self.G(self.F(batch_z), **ae_kwargs)
        fake_images_dicriminator_outputs = self.D(self.E(batch_fake_data, **ae_kwargs))
        loss = F.softplus(-fake_images_dicriminator_outputs).mean()

        return loss

    def get_EG_loss(self, batch_real_data, **ae_kwargs):
        """
        Compute a reconstruction loss in the w latent space for the auto encoder (E,G):
            || F(X) - E(G(F(x))) || = || W - E(G(W)) ||
        """
        batch_z = torch.randn(batch_real_data.shape[0], self.cfg['w_dim'], dtype=torch.float32).to(self.device)
        batch_w = self.F(batch_z)
        batch_reconstructed_w = self.E(self.G(batch_w, **ae_kwargs), **ae_kwargs)
        return torch.mean(((batch_reconstructed_w - batch_w.detach())**2))

    def perform_train_step(self, batch_real_data, tracker, **ae_kwargs):
        """
        Optimizes the model with a batch of real images:
             optimize :Disctriminator, Generator and reconstruction loss of the autoencoder
        """
        # Step I. Update E, and D: optimizer the discriminator D(E( * ))
        self.ED_optimizer.zero_grad()
        L_adv_ED = self.get_ED_loss(batch_real_data, **ae_kwargs)
        L_adv_ED.backward()
        self.ED_optimizer.step()
        tracker.update(dict(L_adv_ED=L_adv_ED))

        # Step II. Update F, and G: Optimize the generator G(F( * )) to fool D(E ( * ))
        self.FG_optimizer.zero_grad()
        L_adv_FG = self.get_FG_loss(batch_real_data, **ae_kwargs)
        L_adv_FG.backward()
        self.FG_optimizer.step()
        tracker.update(dict(L_adv_FG=L_adv_FG))

        # Step III. Update E, and G: Optimize the reconstruction loss in the Latent space W
        self.ED_optimizer.zero_grad()
        self.FG_optimizer.zero_grad()
        # self.EG_optimizer.zero_grad()
        L_err_EG = self.get_EG_loss(batch_real_data, **ae_kwargs)
        L_err_EG.backward()
        # self.EG_optimizer.step()
        self.ED_optimizer.step()
        self.FG_optimizer.step()
        tracker.update(dict(L_err_EG=L_err_EG))

    def train(self, train_dataset, test_data, output_dir):
        raise NotImplementedError

    def generate(self, z_vectors, **ae_kwargs):
        raise NotImplementedError

    def encode(self, img, **ae_kwargs):
        raise NotImplementedError

    def decode(self, latent_vectorsz, **ae_kwargs):
        raise NotImplementedError

    def save_sample(self, dump_path, samples_z, samples, **ae_kwargs):
        """
        Create debug image of images and their reconstruction alongside images generated from random noise
        """
        with torch.no_grad():
            restored_image = self.decode(self.encode(samples, **ae_kwargs), **ae_kwargs)
            generated_images = self.generate(samples_z, **ae_kwargs)

            resultsample = torch.cat([samples, restored_image, generated_images], dim=0).cpu()

            # Normalize images from -1,1 to 0, 1.
            # Eventhough train samples are in this range (-1,1), the generated image may not. But this should diminish as
            # raining continues or else the discriminator can detect them. Anyway save_image clamps it to 0,1
            resultsample = resultsample * 0.5 + 0.5

            save_image(resultsample, dump_path, nrow=len(samples))


class MLP_ALAE(ALAE):
    """
    Implements the MLP version of ALAE. all submodules here are composed of MLP layers
    """
    def __init__(self, model_config, device):
        super().__init__(model_config, 'MLP', device)

    def generate(self, z_vectors, **ae_kwargs):
        return self.G(self.F(z_vectors))

    def encode(self, img, **ae_kwargs):
        return self.E(img)

    def decode(self, latent_vectors, **ae_kwargs):
        return self.G(latent_vectors)

    def train(self, train_dataset, test_data, output_dir):
        train_dataloader = get_dataloader(train_dataset, self.cfg['batch_size'], resize=None, device=self.device)
        tracker = LossTracker(output_dir)
        self.set_optimizers_lr(self.cfg['lr'])
        for epoch in range(self.cfg['epochs']):
            for batch_real_data in tqdm(train_dataloader):
                self.perform_train_step(batch_real_data, tracker)

            tracker.plot()
            dump_path = os.path.join(output_dir, 'images', f"epoch-{epoch}.jpg")
            self.save_sample(dump_path, test_data[0], test_data[1])

            self.save_train_state(os.path.join(output_dir, "last_ckp.pth"))

    def load_train_state(self, checkpoint_path):
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.F.load_state_dict(checkpoint['F'])
            self.G.load_state_dict(checkpoint['G'])
            self.E.load_state_dict(checkpoint['E'])
            self.D.load_state_dict(checkpoint['D'])
            print(f"Checkpoint {os.path.basename(checkpoint_path)} loaded.")

    def save_train_state(self, save_path):
        torch.save(
            {
                'F': self.F.state_dict(),
                'G': self.G.state_dict(),
                'E': self.E.state_dict(),
                'D': self.D.state_dict(),
            },
            save_path
        )


class ImgLoader:
    def __init__(self, center_crop_size, resize, normalize, to_torch, dtype):
        self.center_crop_size = center_crop_size
        self.resize = resize
        self.normalize = normalize
        self.dtype = dtype
        self.to_torch = to_torch

    def __call__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.center_crop_size:
            img = center_crop(img, self.center_crop_size)
        if self.resize:
            img = cv2.resize(img, (self.resize, self.resize))
        img = img.transpose(2, 0, 1)
        if self.normalize:
            img = img / 127.5 - 1
        if self.to_torch:
            img = torch.tensor(img, dtype=self.dtype)
        else:
            img = img.astype(self.dtype)
        return img


def center_crop(img, size):
    y_start = int((img.shape[0] - size)/2)
    x_start = int((img.shape[1] - size)/2)
    return img[y_start: y_start + size, x_start: x_start + size]


def download_mnist(data_dir):
    """
    Taken from torchvision.datasets.mnist
    Dwonloads Mnist  from the official site
    reshapes themas images, normalizes them and saves them as a tensor
    """
    raw_folder = os.path.join(data_dir, 'raw')
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder, exist_ok=True)

        # download files
        train_imgs_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"
        test_imgs_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"
        for url, md5 in [train_imgs_url, test_imgs_url]:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=raw_folder, filename=filename, md5=md5)

    if not os.path.exists(os.path.join(data_dir, 'train_data.pt')):

        # process and save as torch files
        print('Processing...')

        training_set = read_image_file(os.path.join(raw_folder, 'train-images-idx3-ubyte'))
        test_set = read_image_file(os.path.join(raw_folder, 't10k-images-idx3-ubyte'))

        # preprocess: reshape and normalize from [0,255] to [-1,1]
        training_set = training_set.reshape(-1, 1, MNIST_WORKING_DIM, MNIST_WORKING_DIM) / 127.5 - 1
        test_set = test_set.reshape(-1, 1, MNIST_WORKING_DIM, MNIST_WORKING_DIM) / 127.5 - 1

        with open(os.path.join(data_dir, 'train_data.pt'), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(data_dir, 'test_data.pt'), 'wb') as f:
            torch.save(test_set, f)

    print('Done!')


def download_lwf(data_dir):
    """
    Dwonloads LFW alligned images (deep funneled version) from the official site
    crops and normalizes them and saves them as a tensor
    """

    if not os.path.exists(os.path.join(data_dir, 'lfw-deepfunneled.tgz')):
        print("Downloadint LFW from official site...")
        download_and_extract_archive("http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz",
                                     md5='68331da3eb755a505a502b5aacb3c201',
                                     download_root=data_dir, filename='lfw-deepfunneled.tgz')
    if not os.path.exists(os.path.join(data_dir, 'lfw-deepfunneled')):
        f = tarfile.open(os.path.join(data_dir, 'lfw-deepfunneled.tgz'), 'r:gz')
        f.extractall(data_dir)
        f.close()


def download_celeba(data_dir):
    print("Downloading Celeb-a from kaggle...")
    os.environ['KAGGLE_USERNAME'] = "ariel415el"
    os.environ['KAGGLE_KEY'] = "831db7b1693cd81d31ce16e340ddba03"
    import kaggle
    kaggle.api.dataset_download_files('jessicali9530/celeba-dataset', path=data_dir, unzip=True, quiet=False)
    print("Done!")

def download_ffhq_thumbnails(data_dir):
    print("Downloadint FFHQ-thumbnails from kaggle...")
    os.environ['KAGGLE_USERNAME'] = "ariel415el"
    os.environ['KAGGLE_KEY'] = "831db7b1693cd81d31ce16e340ddba03"
    import kaggle
    kaggle.api.dataset_download_files('greatgamedota/ffhq-face-data-set', path=data_dir, unzip=True, quiet=False)
    print("Done.")


def get_lfw(data_dir, dim):
    """
    Returns an LFW train and val datalsets
    """
    download_lwf(data_dir)
    pt_name = f"LFW-{dim}x{dim}.pt"
    if not os.path.exists(os.path.join(data_dir, pt_name)):
        print("Preprocessing FFHQ data")
        imgs = []
        img_loader = ImgLoader(center_crop_size=150, resize=dim, normalize=True, to_torch=False, dtype=np.float32)
        for celeb_name in tqdm(os.listdir(os.path.join(data_dir, 'lfw-deepfunneled'))):
            for fname in os.listdir(os.path.join(data_dir, 'lfw-deepfunneled', celeb_name)):
                img = img_loader(os.path.join(data_dir, 'lfw-deepfunneled', celeb_name, fname))
                imgs.append(torch.tensor(img, dtype=torch.float32))
        with open(os.path.join(data_dir, pt_name), 'wb') as f:
            torch.save(torch.stack(imgs), f)

    data = torch.load(os.path.join(data_dir, pt_name))

    dataset = MemoryDataset(data)
    val_size = int(len(dataset) * VAL_SET_PORTION)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])

    return train_dataset, val_dataset


def get_mnist(data_dir):
    """
    Returns an LFW train and val datalsets
    """
    download_mnist(data_dir)
    train_data = torch.load(os.path.join(data_dir, "train_data.pt"))
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"))
    train_dataset, val_dataset = MemoryDataset(train_data), MemoryDataset(test_data)

    return train_dataset, val_dataset, MNIST_WORKING_DIM


def get_celeba(data_dir, dim):
    imgs_dir = os.path.join(data_dir, 'img_align_celeba', 'img_align_celeba')
    if not os.path.exists(imgs_dir):
        download_celeba(data_dir)
    img_loader = ImgLoader(center_crop_size=170, resize=dim, normalize=True, to_torch=True, dtype=torch.float32)
    img_paths = [os.path.join(imgs_dir, fname) for fname in os.listdir(imgs_dir)]
    dataset = DiskDataset(img_paths, img_loader)
    val_size = int(len(dataset) * VAL_SET_PORTION)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])

    return train_dataset, val_dataset

def get_ffhq(data_dir, dim):
    imgs_dir = os.path.join(data_dir, 'thumbnails128x128')
    if not os.path.exists(imgs_dir):
        download_ffhq_thumbnails(data_dir)

    if dim <= 64:
        pt_file = f"FFHQ_Thumbnail-{dim}x{dim}.pt"
        if not os.path.exists(os.path.join(data_dir, pt_file)):
            print(f"Preprocessing FFHQ: creating a {dim}x{dim}  version of all data")
            imgs = []
            img_loader = ImgLoader(center_crop_size=None, resize=dim, normalize=True, to_torch=True, dtype=torch.float32)
            for img_name in tqdm(os.listdir(imgs_dir)):
                fname = os.path.join(imgs_dir, img_name)
                img = img_loader(fname)
                imgs.append(img)
            with open(os.path.join(data_dir, pt_file), 'wb') as f:
                torch.save(torch.stack(imgs), f)

        data = torch.load(os.path.join(data_dir, pt_file))
        dataset = MemoryDataset(data)
    else:
        img_loader = ImgLoader(center_crop_size=None, resize=dim, normalize=True, to_torch=True, dtype=torch.float32)
        img_paths = [os.path.join(imgs_dir, img_name) for img_name in os.listdir(imgs_dir)]
        dataset = DiskDataset(img_paths, img_loader)
    val_size = int(len(dataset) * VAL_SET_PORTION)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])

    return train_dataset, val_dataset

class MemoryDataset(Dataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx):
        return self.data_matrix[idx]

    def get_data(self):
        return self.data_matrix


class DiskDataset(Dataset):
    def __init__(self, image_paths, load_image_function):
        self.image_paths = image_paths
        self.load_image_function = load_image_function

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.load_image_function(self.image_paths[idx])


class EndlessDataloader:
    """
    An iterator wrapper for a dataloader that resets when reaches its end
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        try:
            real_image = next(self.iterator)

        except (OSError, StopIteration):
            self.iterator = iter(self.dataloader)
            real_image = next(self.iterator)

        return real_image


def get_dataset(data_root, dataset_name, dim):
    if dataset_name.lower() == 'mnist':
        assert dim == 28
        train_dataset, test_dataset, _ = get_mnist(os.path.join(data_root, 'Mnist'))
    elif dataset_name.lower() == 'celeb-a':
        train_dataset, test_dataset = get_celeba(os.path.join(data_root, 'Celeb-a'), dim)
    elif dataset_name.lower() == 'ffhq':
        train_dataset, test_dataset = get_ffhq(os.path.join(data_root, 'FFHQ-thumbnails'), dim)
    elif dataset_name.lower() == 'lfw':
        train_dataset, test_dataset = get_lfw(os.path.join(data_root, 'LFW'), dim)

    else:
        raise ValueError("No such available dataset")

    return train_dataset, test_dataset


class RequireGradCollator(object):
    def __init__(self, resize, device):
        self.device = device
        self.resize = resize

    def __call__(self, batch):
        with torch.no_grad():
            # requires_grad=True is necessary for the gradient penalty calculation
            # return torch.tensor(batch, requires_grad=True, device=self.device, dtype=torch.float32)
            batch_tensor = torch.stack(batch).to(self.device).float()
            if self.resize is not None:
                batch_tensor = torch.nn.functional.interpolate(batch_tensor, (self.resize, self.resize))
            batch_tensor.requires_grad = True
            return batch_tensor


def get_dataloader(dataset, batch_size, resize, device):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'collate_fn': RequireGradCollator(resize, device)}
    if device == "cuda:0":
        kwargs.update({'num_workers': 2,
                       'pin_memory': True})
    return torch.utils.data.DataLoader(dataset, **kwargs)


parser = argparse.ArgumentParser(description='Train arguments')
parser.add_argument("--output_root", type=str, default="Training_dir-test")
parser.add_argument("--num_debug_images", type=int, default=24)
parser.add_argument("--print_model", action='store_true', default=False)
parser.add_argument("--print_config", action='store_true', default=False)
parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0/cpu")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "cuda:0" else "cpu")

config = {'z_dim':50,
          'w_dim':50,
          'mapping_layers': 6,
          'image_dim':28,
          'lr': 0.002,
          "batch_size": 128,
          'epochs':100}

if __name__ == '__main__':
    config_descriptor = get_config_str(config)
    output_dir = os.path.join(args.output_root, f"MlpALAE_d-Mnist_{config_descriptor}")
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    # create_dataset
    train_dataset, test_dataset = get_dataset("data", "Mnist", dim=config['image_dim'])

    if args.print_config:
        print("Model config:")
        pprint(config)

    # Create model
    model = MLP_ALAE(model_config=config, device=device)
    if args.print_model:
        print(model)

    test_dataloader = get_dataloader(test_dataset, batch_size=args.num_debug_images, resize=None, device=device)
    test_samples_z = torch.randn(args.num_debug_images, config['z_dim'], dtype=torch.float32).to(device)
    test_samples = next(iter(test_dataloader))

    ckp_path = os.path.join(output_dir, "last_ckp.pth")
    if os.path.exists(ckp_path):
        print("Playing with trained model")
        model.load_train_state(ckp_path)
        N = min(8, args.num_debug_images // 2)
        plot_latent_interpolation(model, test_samples[:N], test_samples[N: 2*N], steps=5, plot_path=os.path.join(output_dir, "linear_inerpolation.png"))

    else:
        print("Training model")
        model.train(train_dataset, (test_samples_z, test_samples), output_dir)