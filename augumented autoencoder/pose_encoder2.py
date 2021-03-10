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



def onehot(x, num_classes=0):
    if num_classes == 0:
        num_classes = x.max()+1
    return np.eye(num_classes)[x.flatten()].reshape(*x.shape,-1)

class DB(Dataset):
    def __init__(self, args, img_path): #targetは使用しないので削除
        self.imgsz = args.imgsz
        self.num_classes = args.num_classes

        self.images = self.getPath(img_path)
        #self.targets = self.getPath(target_path)

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

    def getImage(self, path):    #
        #path = self.images[index]
        #print("load path is ",path)
        pick_name = path.replace("./datasets/AAE_pose/SSD_AR0_30/","")
        # print("pick_name",pick_name)
        pick_name = pick_name.replace(".png","")#保存名の取り出し
        #print("pick_name",pick_name)
        sample = skimage.io.imread(path)
        sample = skimage.transform.resize(sample, (self.imgsz, self.imgsz), \
                mode='reflect', anti_aliasing=True)

        sample = sample.reshape(sample.shape[0], sample.shape[1], 3)
        # change H,W,C to C,H,W
        sample = sample.transpose((2, 0, 1))

        if np.issubdtype(sample.dtype, np.integer):
            sample = sample/255.
            #sample = 2.0*sample - 1.0

        return torch.Tensor(sample), pick_name

    def __getitem__(self, index):
        images,pick_name = self.getImage(self.images[index])
        #print("get_item pickname ",pick_name)
        #targets = self.getImage(self.targets[index])
        return images, pick_name

    def __len__(self):
        return len(self.images)

def main(args):
    print("check check")
    print(args)


    #torch.manual_seed(22)
    #np.random.seed(22)
    #data 22740
    #batch size is 100
    # 22740 / 100 = 227 ... 1
    db_pose = DB(args, img_path=args.pose)
    print("db_pose is ",db_pose)
    poseloader = DataLoader(db_pose, batch_size=args.batchsz, shuffle=False, \
            num_workers=8, pin_memory=True)
    # print("main pick name is ",pick_name)
    #modelの定義
    device = torch.device('cuda')
    #pose =  .readline()
    aae = AAE(args).to(device)
    optimizer = optim.Adam(aae.parameters(), lr=args.lr)

    #modelの読み込みckpファイルを用いて
    aae.load_state_dict(torch.load(args.load))
    print('load ckpt from:', args.load)




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

    # last_loss, last_ckpt, last_disp = 0, 0, 0
    # #i = len(db_loader)
    # time_data, time_vis = 0, 0
    # time_start = time.time()
    
    # 評価モード
    #aae.eval()
    #AE_sum     = 0.0
    print("poseloader length is ",len(poseloader)) #poseloader is 228
    with torch.no_grad():
        for batch in poseloader:
            x = batch[0].to(device, dtype=torch.float, non_blocking=True)
            #print("torch",x.dtype)
            name = batch[1]
            #kokonihaittekuru
            #おそらくxそのままではできない
            #pillowで扱える形式に変換するのを挟む
            # x_numpy = x.to('cpu').detach().numpy().copy()
            # result = x_numpy[0,:, :, :]
            # pil_img = Image.fromarray((result[0]*255).astype(np.uint8))
            # for u in range(0,360,4):
            #     w, h = pil_img.size
            #     rotated_image = Image.new('RGB',(3*w,3*h), color=(255,255,255))
            #     rotated_image.paste(pil_img, (w, h))
            #     rotated_image = rotated_image.rotate(angle=u*-1, resample=Image.BICUBIC, expand=False)
            #     rotated_image = rotated_image.crop((w,h,2*w,2*h))
            #     #print("rotated_image",rotated_image.shape)

            #     #もしかしてこれでできるかも．．．
            #     #rotated_image = rotated_image.transpose((2, 0, 1))

            #     roll_pi = "r"+str(u)

            #     im = np.array(rotated_image)
            #     im = im[:, :, :, np.newaxis]
            #     im = im.transpose((3,2, 0, 1))
            #     im = im.astype(np.float32)
            #     x_roll = torch.from_numpy(im).clone()
            #     x_roll = x_roll.cuda()

            # #pillowをencoderが扱える形式にする



            #print("name is ",len(name))
            
            #target = x[1].to(device, dtype=torch.float, non_blocking=True)
            z = aae.encoder(x) # z is 潜在変数は１２８
            #print("z length", len(z))
            #潜在変数をtxtに書き出したい
            #txtは22470個出す
            z = z.cpu()
            for i in range(len(z)): #100回
                for n in range(128):  #128回
                    latent_variable = str(z[i][n])  #書き込みたい数字
    
                    latent_variable = latent_variable.replace("tensor(", "")
                    latent_variable = latent_variable.replace(")","")
                    save_name = name[i] #./dataset/hyouka/
                    # print("bef name",save_name)
                    save_name = save_name.replace("./datasets/AAE_pose/SSD_AR0_30/","")

                    with open(args.name+ "/SSD_AR0_30_last/" +save_name+".txt","a") as f:
                        f.write(latent_variable + "\n")

            
            #AE_sum += z* batchsz



    #     j = len(poseloader)
    #     with open(args.name+'/latent_variable_pose.txt', "a") as f:
    #         f.write(str(args.load)+str(args.beta*AE_sum/j) + "\n")

    #     x, xr = [img[:8].cpu() for img in (x, xr)]
    #     x, xr = [img.clamp(0, 1) for img in (x, xr)]

    #     # save images
    #     save_image(torch.cat([x,xr], 0), \
    #             args.name+'/test/x_xr_%010d.png' % epoch_num, nrow=4)


    # torch.save(aae.state_dict(), args.name+'/ckpt/aae_%010d.mdl'%epoch_num)
    # print('saved final ckpt:', args.name+'/ckpt/aae_%010d.mdl'%epoch_num)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--imgsz', type=int, default=128, \
            help='imgsz')
    argparser.add_argument('--batchsz', type=int, default=1, \
            help='batch size')
    argparser.add_argument('--z_dim', type=int, default=128, \
            help='hidden latent z dim')
    argparser.add_argument('--epoch', type=int, default=500, \
            help='epochs to train')
    argparser.add_argument('--beta', type=float, default=1.0, \
            help='beta * ae_loss')
    argparser.add_argument('--lr', type=float, default=0.0002, \
            help='learning rate')

    argparser.add_argument('--load', type=str, required=True, \
            help='checkpoint to load')
    argparser.add_argument('--pose', type=str, default='data', \
            help='root/label/*.jpg')
    argparser.add_argument('--resume', type=str, default=None, \
            help='with ckpt path, set empty str to load latest ckpt')
    argparser.add_argument('--name', type=str, default='aae', \
            help='name for storage and checkpoint')
    argparser.add_argument('--num_classes', type=int, default=-1, \
            help='set to positive value to model shapes (e.g. segmentation)')


    args = argparser.parse_args()
    main(args)
