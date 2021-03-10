import time
import numpy as np
import random

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResBlk(nn.Module):
    def __init__(self, kernels, chs):
        """
        :param kernels: [1, 3, 3], as [kernel_1, kernel_2, kernel_3]
        :param chs: [ch_in, 64, 64, 64], as [ch_in, ch_out1, ch_out2, ch_out3]
        :return:
        """
        assert len(chs)-1 == len(kernels), "mismatching between chs and kernels"
        assert all(map(lambda x: x%2==1, kernels)), "odd kernel size only"
        super(ResBlk, self).__init__()
        layers = []
        for idx in range(len(kernels)):
            layers += [nn.Conv2d(chs[idx], chs[idx+1], kernels[idx], \
                        padding = kernels[idx]//2), \
                        nn.LeakyReLU(0.2, True)]
        layers.pop() # remove last activation
        self.net = nn.Sequential(*layers)
        self.shortcut = nn.Sequential()
        if chs[0] != chs[-1]: # convert from ch_int to ch_out3
            self.shortcut = nn.Conv2d(chs[0], chs[-1], kernel_size=1)
        self.outAct = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.outAct(self.shortcut(x) + self.net(x))

class Encoder(nn.Module):
    def __init__(self, imgsz, ch, z_dim, io_ch=3):
        """
        :param imgsz:
        :param ch: base channels
        :param z_dim: latent space dim
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential( \
                nn.Conv2d(io_ch, ch, kernel_size=5, stride=1, padding=2),
                nn.LeakyReLU(0.2, True),
                nn.AvgPool2d(2, stride=None, padding=0)))

        # [b, ch_cur, imgsz, imgsz] => [b, ch_next, mapsz, mapsz]
        mapsz = imgsz // 2
        ch_cur = ch
        ch_next = ch_cur * 2
        while mapsz > 8: # util [b, ch_, 8, 8]
            # add resblk
            self.layers.append(nn.Sequential( \
                    ResBlk([1, 3, 3], [ch_cur]+[ch_next]*3), \
                    nn.AvgPool2d(kernel_size=2, stride=None)))
            mapsz = mapsz // 2
            ch_cur = ch_next
            ch_next = ch_next * 2 if ch_next < 512 else 512 # set max ch=512

        # 8*8 -> 4*4
        self.layers.append(nn.Sequential( \
                ResBlk([3, 3], [ch_cur, ch_next, ch_next]), \
                nn.AvgPool2d(kernel_size=2, stride=None)))
        mapsz = mapsz // 2

        # 4*4 -> 4*4
        self.layers.append(nn.Sequential( \
                ResBlk([3, 3], [ch_next, ch_next, ch_next])))

        self.z_net = nn.Linear(ch_next*mapsz*mapsz, z_dim)

        # just for print
        x = torch.randn(2, io_ch, imgsz, imgsz)
        print('Encoder:', list(x.shape), end='=>')
        with torch.no_grad():
            for layer in self.layers[:-1]:
                x = layer(x)
                print(list(x.shape), end='=>')
            x = self.layers[-1](x)
            x = x.view(x.shape[0], -1)
            print(list(x.shape), end='=>')
            x = self.z_net(x)
            print(list(x.shape), end='=>')
            #print(list(x[0].shape), list(x[1].shape))
        #print(self.layers)
        #print(self.z_net)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.shape[0], -1)
        z = self.z_net(x)
        return z

class Decoder(nn.Module):
    def __init__(self, imgsz, ch, z_dim, io_ch=3):
        """
        :param imgsz:
        :param ch: base channels
        :param z_dim: latent space dim
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.insert(0, nn.Sequential( \
                nn.Conv2d(ch, io_ch, kernel_size=5, stride=1, padding=2)))

        self.layers.insert(0, nn.Sequential( \
                nn.Upsample(scale_factor=2),
                ResBlk([3, 3], [ch, ch, ch])))

        mapsz = imgsz // 2
        ch_cur = ch
        ch_next = ch_cur * 2
        while mapsz > 16: # util [b, ch_, 16, 16]
            self.layers.insert(0, nn.Sequential( \
                    nn.Upsample(scale_factor=2),
                    ResBlk([1, 3, 3], [ch_next]+[ch_cur]*3)))
            mapsz = mapsz // 2
            ch_cur = ch_next
            ch_next = ch_next * 2 if ch_next < 512 else 512 # set max ch=512

        # 16*16, 8*8
        for _ in range(2):
            self.layers.insert(0, nn.Sequential( \
                    nn.Upsample(scale_factor=2),
                    ResBlk([3, 3], [ch_next]+[ch_cur]*2)))
            mapsz = mapsz // 2
            ch_cur = ch_next
            ch_next = ch_next * 2 if ch_next < 512 else 512 # set max ch=512

        # 4*4
        self.layers.insert(0, nn.Sequential( \
                ResBlk([3, 3], [ch_next]+[ch_cur]*2)))

        # fc
        self.z_net = nn.Sequential( \
                nn.Linear(z_dim, ch_next*mapsz*mapsz),
                nn.ReLU(True))

        # just for print
        x = torch.randn(2, z_dim)
        print('Decoder:', list(x.shape), end='=>')
        x = self.z_net(x)
        print(list(x.shape), end='=>')
        with torch.no_grad():
            x = x.view(x.shape[0], -1, 4, 4)
            x = self.layers[0](x)
            for layer in self.layers[1:]:
                print(list(x.shape), end='=>')
                x = layer(x)
            print(list(x.shape))
        #print(self.z_net)
        #print(self.layers)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.z_net(x)
        x = x.view(x.shape[0], -1, 4, 4)
        for layer in self.layers:
            x = layer(x)
        return x

class AAE(nn.Module):
    def __init__(self, args):
        super(AAE, self).__init__()
        imgsz = args.imgsz
        z_dim = args.z_dim

        self.beta = args.beta
        io_ch = 3

        self.encoder = Encoder(imgsz, 16, z_dim, io_ch)
        self.decoder = Decoder(imgsz, 16, z_dim, io_ch)

        self.z_dim = z_dim # z is the hidden vector while h is the output of encoder
        self.optim_encoder = optim.Adam(self.encoder.parameters(), lr=args.lr)
        self.optim_decoder = optim.Adam(self.decoder.parameters(), lr=args.lr)

    def reparam(self, mu, logvar):
        # sample from normal dist
        eps = torch.randn_like(mu)
        # reparameterization trick
        std = torch.exp(0.5*logvar)
        z = mu + std * eps
        return z

    def kld(self, mu, logvar):
        """
        compute the kl divergence between N(mu, std) and N(0, 1)
        :return:
        """
        kl = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()/mu.shape[0]
        return kl

    def forward(self, x, target):
        """
        x : miss_image
        target : original_image
        """

        # forward 1
        z = self.encoder(x)
        xr = self.decoder(z)

        # backward
        ae = (target - xr)**2
        ae = 0.5*ae.view(ae.shape[0],-1).sum(dim=1).mean()
        #ae = F.mse_loss(xr, x, reduction='sum')*0.5/x.shape[0]
        #reg = self.kld(mu, logvar)
        loss = self.beta*ae

        # store
        AE = ae.item()

        return loss, xr.detach(), AE

if __name__ == '__main__':
    Encoder(128)
    Decoder(128)
