import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os, glob
import os.path
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from    torch import optim
import numpy as np
from   torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import argparse
from   torchvision.utils import save_image
from   model2 import AAE

import tqdm
import skimage
import time, random, math
from PIL import Image

import math


def onehot(x, num_classes=0):
    if num_classes == 0:
        num_classes = x.max()+1
    return np.eye(num_classes)[x.flatten()].reshape(*x.shape,-1)

class DB(Dataset):
    def __init__(self, args, img_path, target_path):
        self.imgsz = args.imgsz
        self.num_classes = args.num_classes

        self.images = self.getPath(img_path)
        self.targets = self.getPath(target_path)

    def getPath(self, path):
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', \
                '.bmp', '.pgm', '.tif', '.tiff', '.webp')

        images = []

        for root, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if fname.lower().endswith(extensions):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def getImage(self, path,agree,zoom):    #書き換え
        #path = self.images[index]
        sample = skimage.io.imread(path)
        sample =self.aug(sample,agree,zoom)
        sample = skimage.transform.resize(sample, (self.imgsz, self.imgsz), \
                mode='reflect', anti_aliasing=True)

        sample = sample.reshape(sample.shape[0], sample.shape[1], 3)
        # change H,W,C to C,H,W
        sample = sample.transpose((2, 0, 1))

        if np.issubdtype(sample.dtype, np.integer):
            sample = sample/255.
            #sample = 2.0*sample - 1.0

        return torch.Tensor(sample)

    def aug(self, img,agree,zoom):
        #回転，明度変換など,距離
        # print("img",type(img))
        # print("img shape",img.shape)
        # print("img shape 0",img.shape[0])
        #rotated_image = cv2.imread(img)
        #rotated_image = Image.open(img)
        h, w, c = img.shape
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), agree,zoom)
        affine_img = cv2.warpAffine(img, mat, (w, h), borderValue=(255,255,255))
        #affine_img = cv2.warpAffine(img, mat, (w, h))
        # w, h = img.shape[0], img.shape[1]
        # rotated_image = Image.new('RGB',(3*w,3*h), color=(255,255,255))
        # rotated_image.paste(img, (w, h))
        # rotated_image = rotated_image.rotate(angle=agree, resample=Image.BICUBIC, expand=False)
        # rotated_image = rotated_image.crop((w,h,2*w,2*h))
        return affine_img

    ######

    

    def __getitem__(self, index):
        #image ->a ,target ->b
        # self.1st = random.randint(0, 10)
        # self.2nd = random.randint(80, 100)
        # self.3rd = random.randint(170, 190)
        # self.4th = random.randint(260, 280)
        # self.5th = random.randint(350,360)

        #self.agree = random.choice([0,1,2,3,4,5,6,7,8,9,10,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,170,171,172,173,174,175,176,177,178,179,180,182,181,183,184,185,186,187,188,189,261,262,263,264,265,267,268,269,270,272,271,273,274,275,276,277,278,279,351,352,353,354,355,356,357,358,359])
        self.agree = random.randint(0,360)
        self.zoom = random.uniform(0.93,1.2)
        #self.zoom = random.choice([0.999,0.998,0.98,0.97,0.96,0.99,1,1,1,1,0.98,0.95])
        images = self.getImage(self.images[index],self.agree,self.zoom)
        targets = self.getImage(self.targets[index],self.agree,self.zoom)
        return images, targets

    def __len__(self):
        return len(self.images)

def main(args):
    print(args)

    #torch.manual_seed(22)
    #np.random.seed(22)


    db = DB(args, img_path=args.root, target_path=args.target)
    db_loader = DataLoader(db, batch_size=args.batchsz, shuffle=True, \
            num_workers=8, pin_memory=True)

    db_test = DB(args, img_path=args.test, target_path=args.test_target)
    testloader = DataLoader(db_test, batch_size=args.batchsz, shuffle=False, \
            num_workers=8, pin_memory=True)


    device = torch.device('cuda')
    aae = AAE(args).to(device)
    optimizer = optim.Adam(aae.parameters(), lr=args.lr)


    params = filter(lambda x: x.requires_grad, aae.parameters())
    num = sum(map(lambda x: np.prod(x.shape), params))
    print('Total trainable tensors:', num)

    for path in [args.name, args.name+'/res', args.name+'/ckpt', args.name+'/test']:
        if not os.path.exists(path):
            os.mkdir(path)
            print('mkdir:', path)

    iter_cnt = 0
    if args.resume is not None and args.resume != 'None':
        if args.resume is '': # load latest
            ckpts = glob.glob(args.name+'/ckpt/*_*.mdl')
            if not ckpts:
                print('no avaliable ckpt found.')
                raise FileNotFoundError
            ckpts = sorted(ckpts, key=os.path.getmtime)
            # print(ckpts)
            ckpt = ckpts[-1]
            iter_cnt = int(ckpt.split('.')[-2].split('_')[-1])
            aae.load_state_dict(torch.load(ckpt))
            print('load latest ckpt from:', ckpt, iter_cnt)
        else: # load specific ckpt
            if os.path.isfile(args.resume):
                aae.load_state_dict(torch.load(args.resume))
                print('load ckpt from:', args.resume, iter_cnt)
            else:
                raise FileNotFoundError
    else:
        print('training from scratch...')

    # training.
    print('>>training AAE now...')

    last_loss, last_ckpt, last_disp = 0, 0, 0
    i = len(db_loader)
    time_data, time_vis = 0, 0
    time_start = time.time()
    for _ in tqdm.trange(args.epoch, desc='epoch'):
        tqdm_iter = tqdm.tqdm(db_loader, desc='iter', \
                bar_format=str(args.batchsz)+': {n_fmt}/{total_fmt}'+\
                '[{elapsed}<{remaining},{rate_fmt}]'+'{postfix}')

        AE_sum = 0.0
        #import pdb; pdb.set_trace()

        for batch in tqdm_iter:
            time_data = time.time() - time_start
            #import pdb; pdb.set_trace()
            iter_cnt += 1
            x = batch[0].to(device, dtype=torch.float, non_blocking=True)
            target = batch[1].to(device, dtype=torch.float, non_blocking=True)
            loss, xr, AE = aae(x, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            nan_num = 0
            for var in (xr, AE):
                if not isinstance(var, torch.Tensor): var = torch.tensor(var)
                assert not torch.isnan(var).any(), '[{0}]nan detected!'.format(nan_num)
                nan_num +=1

            AE_sum += AE

            time_start = time.time()
            last_loss = iter_cnt
            if iter_cnt % i == 0:
                last_disp = iter_cnt
                epoch_num = iter_cnt / i
                target,x, xr = [img[:8].cpu() for img in (target,x, xr)]
                # display images

                target,x, xr = [img.clamp(0, 1) for img in (target,x, xr)]
                # save images
                save_image(torch.cat([target,x,xr], 0), \
                        args.name+'/res/target_x_xr_%010d.png' % epoch_num, nrow=4)

            if iter_cnt % (i*5) == 0:
                last_ckpt = iter_cnt
                # save checkpoint
                torch.save(aae.state_dict(), \
                        args.name+'/ckpt/aae_%010d.mdl'%epoch_num)

            time_vis = time.time() - time_start
            time_start = time.time()
            postfix = '[%d/%d/%d/%d]'%( \
                    iter_cnt, last_loss, last_disp, last_ckpt)
            if time_data >= 0.1:
                postfix += ' data %.1f'%time_data
            if time_vis >= 0.1:
                postfix += ' vis %.1f'%time_vis
            tqdm_iter.set_postfix_str(postfix)

        epoch_num = iter_cnt / i

        with open(args.name+'/aae_loss_val.txt', "a") as f:
            f.write(str(args.beta*AE_sum/i) + "\n")

        # 評価モード
        #aae.eval()
        AE_sum     = 0.0

        with torch.no_grad():
            for batch in testloader:
                x = batch[0].to(device, dtype=torch.float, non_blocking=True)
                target = batch[1].to(device, dtype=torch.float, non_blocking=True)
                loss, xr, AE = aae(x, target)

                AE_sum   += AE

        j = len(testloader)
        with open(args.name+'/loss_val_test.txt', "a") as f:
            f.write(str(args.beta*AE_sum/j) + "\n")

        target,x, xr = [img[:8].cpu() for img in (target,x, xr)]
        target,x, xr = [img.clamp(0, 1) for img in (target,x, xr)]

        # save images
        save_image(torch.cat([target,x,xr], 0), \
                args.name+'/test/x_xr_%010d.png' % epoch_num, nrow=4)


    torch.save(aae.state_dict(), args.name+'/ckpt/aae_%010d.mdl'%epoch_num)
    print('saved final ckpt:', args.name+'/ckpt/aae_%010d.mdl'%epoch_num)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, default=128, \
            help='imgsz')
    argparser.add_argument('--batchsz', type=int, default=100, \
            help='batch size')
    argparser.add_argument('--z_dim', type=int, default=128, \
            help='hidden latent z dim')
    argparser.add_argument('--epoch', type=int, default=1000, \
            help='epochs to train')
    argparser.add_argument('--beta', type=float, default=1.0, \
            help='beta * ae_loss')
    argparser.add_argument('--lr', type=float, default=0.0002, \
            help='learning rate')
    argparser.add_argument('--root', type=str, default='data', \
            help='root/label/*.jpg')
    argparser.add_argument('--target', type=str, default='data', \
            help='root/label/*.jpg')

    argparser.add_argument('--test', type=str, default='data', \
            help='root/label/*.jpg')
    argparser.add_argument('--test_target', type=str, default='data', \
            help='root/label/*.jpg')
    argparser.add_argument('--resume', type=str, default=None, \
            help='with ckpt path, set empty str to load latest ckpt')

    argparser.add_argument('--name', type=str, default='aae', \
            help='name for storage and checkpoint')
    argparser.add_argument('--num_classes', type=int, default=-1, \
            help='set to positive value to model shapes (e.g. segmentation)')


    args = argparser.parse_args()
    main(args)

