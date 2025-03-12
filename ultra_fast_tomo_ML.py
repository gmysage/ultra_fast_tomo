import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import time
import h5py
import torch.optim as optim

import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
import torchvision
from numba import jit, njit, prange
from skimage import io

from scipy.ndimage import geometric_transform
from numpy import sin, cos
from skimage.transform import warp



def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_3C(nn.Module):
    def __init__(self, nf=32, gc=16, bias=True):
        super(ResidualDenseBlock_3C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        #self.conv11 = nn.Conv2d(nf, int(gc/2), 3, stride=1, dialation=1, padding='same', bias=bias)
        self.conv1 = nn.Conv2d(nf, gc, 3, stride=1, padding='same', bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, stride=1, padding='same', bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, 3, 1, 1, bias=bias)
        #self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        #self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        s = {}
        #x11 = self.lrelu(self.conv11(x))
        #x12 = self.lrelu(self.conv12(x))
        #x1 = torch.cat((x11, x12), 1)
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        #x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        #x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x3 * 0.4 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_3C(nf, gc)
        self.RDB2 = ResidualDenseBlock_3C(nf, gc)
        self.RDB3 = ResidualDenseBlock_3C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first_1 = nn.Conv2d(in_nc, int(nf/2), 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(in_nc, int(nf/2), 3, stride=1, dilation=1, padding='same', bias=True)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        #self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        #x1 = self.conv_first_1(x)
        #x2 = self.conv_first_2(x)
        #fea = torch.cat((x1, x2), 1)
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        #fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        #fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out

class SRCNNDataset(Dataset):
    def __init__(self, image_data, labels):
        self.image_data = image_data
        self.labels = labels

    def __len__(self):
        return (len(self.image_data))

    def __getitem__(self, index):
        image = self.image_data[index]
        label = self.labels[index]
        return (torch.tensor(image, dtype=torch.float),
                torch.tensor(label, dtype=torch.float)
               )

import glob
from skimage import io
class tomoDataset(Dataset):
    def __init__(self, blur_dir, gt_dir, length=None, transform=None):
        super().__init__()
        self.fn_blur = np.sort(glob.glob(f'{blur_dir}/*'))
        self.fn_gt = np.sort(glob.glob(f'{gt_dir}/*'))
        self.transform = transform
        if not length is None:
            self.fn_blur = self.fn_blur[:length]
            self.fn_gt = self.fn_gt[:length]

    def __len__(self):
        return len(self.fn_blur)

    def __getitem__(self, idx):
        img_blur = io.imread(self.fn_blur[idx])
        img_blur = np.expand_dims(img_blur, axis=0)
        img_blur = torch.tensor(img_blur, dtype=torch.float)
        img_gt = io.imread(self.fn_gt[idx])
        img_gt = np.expand_dims(img_gt, axis=0)
        img_gt = torch.tensor(img_gt, dtype=torch.float)
        if self.transform:
            img_blur = self.transform(img_blur, dtype=torch.float)
            img_gt = self.transform(img_gt, dtype=torch.float)
        return img_blur, img_gt

def psnr(label, outputs, max_val=1):
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    max_val = max(np.max(label), np.max(outputs))
    img_diff = (outputs - label) / max_val
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(1. / rmse)
        return PSNR


def get_features_vgg19(image, model_feature, layers=None):
    if layers is None:
        layers = {'2': 'conv1_2',
                  '7': 'conv2_2',
                  '16': 'conv3_4',
                  '25': 'conv4_4'
                 }
    features = {}
    x = image
    for idx, layer in enumerate(model_feature):
        x = layer(x)
        if str(idx) in layers:
            features[layers[str(idx)]] = x
    return features

def vgg_loss(outputs, label, model_feature=[], device='cuda'):
    if not torch.is_tensor(outputs):
        out = torch.tensor(outputs)
    else:
        out = outputs.clone().detach()
    if not torch.is_tensor(label):
        lab = torch.tensor(label).detach()
    else:
        lab = label.clone()
    lab_max = torch.max(lab)
    out = out / lab_max
    lab = lab / lab_max
    if out.shape[1] == 1:
        out = out.repeat(1,3,1,1)
    if lab.shape[1] == 1:
        lab = lab.repeat(1,3,1,1)
    out = out.to(device)
    lab = lab.to(device)

    feature_out1 = 0.5*get_features_vgg19(out, vgg19, {'2': 'conv1_2'})['conv1_2']
    feature_out2 = 0.5*get_features_vgg19(out, vgg19, {'25': 'conv4_4'})['conv4_4']
    feature_lab1 = 0.5*get_features_vgg19(lab, vgg19, {'2': 'conv1_2'})['conv1_2']
    feature_lab2 = 0.5*get_features_vgg19(lab, vgg19, {'25': 'conv4_4'})['conv4_4']
    feature_loss = nn.MSELoss()(feature_out1, feature_lab1) + nn.MSELoss()(feature_out2, feature_lab2)
    return feature_loss

def l1_loss(inputs, targets):
    loss = nn.L1Loss()
    output = loss(inputs, targets)
    return output

def fft_loss(inputs, targets):
    fft_1 = torch.fft.fft2(inputs)
    fft_2 = torch.fft.fft2(targets)
    #fft_1 = torch.log(torch.abs(fft_1))
    #fft_2 = torch.log(torch.abs(fft_2))
    fft_1 = fft_1.imag
    fft_2 = fft_2.imag
    output = nn.MSELoss()(fft_1, fft_2)
    return output

def tv_loss(c):
    x = c[:,:,1:,:] - c[:,:,:-1,:]
    y = c[:,:,:,1:] - c[:,:,:,:-1]
    loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
    return loss

def train(model, dataloader, feature_loss_r=0.005, tv_loss_r=1, l1_loss_r=1, fft_loss_r=1, device='cuda'):
    model.train()
    running_loss = 0.0
    running_feature_loss = 0.0
    running_tv_loss = 0.0
    running_l1_loss = 0.0
    running_fft_loss = 0.0
    running_psnr = 0.0
    total_var_loss = 0.0
    final_loss = 0.0
    final_feature_loss = 0.0
    final_l1_loss = 0.0
    final_fft_loss = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
        image_data = data[0].to(device)
        label = data[1].to(device)

        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)
        feature_loss = vgg_loss(outputs, label).detach() * feature_loss_r
        total_var_loss = tv_loss(outputs).detach() * tv_loss_r
        l_loss = l1_loss(outputs, label).detach() * l1_loss_r
        f_loss = fft_loss(outputs, label).detach() * fft_loss_r
        total_loss = loss + total_var_loss + l_loss + f_loss
        # backpropagation
        total_loss.backward()
        # update the parameters
        optimizer.step()
        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        running_feature_loss += feature_loss.item()
        running_tv_loss += total_var_loss.item()
        running_l1_loss += l_loss.item()
        running_fft_loss += f_loss.item()
        # calculate batch psnr (once every `batch_size` iterations)
        batch_psnr =  psnr(label, outputs)
        running_psnr += batch_psnr
    final_loss = running_loss/len(dataloader.dataset)
    final_feature_loss = running_feature_loss/len(dataloader.dataset)
    #final_feature_loss = 0
    final_tv_loss = running_tv_loss/len(dataloader.dataset)
    final_l1_loss = running_l1_loss/len(dataloader.dataset)
    final_fft_loss = running_fft_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(dataloader.dataset)/dataloader.batch_size)
    return final_loss, final_psnr, final_feature_loss, final_tv_loss, final_l1_loss, final_fft_loss


def validate(model, dataloader, epoch, feature_loss_r=0.005, tv_loss_r=1, l1_loss_r=1, fft_loss_r=1, device='cuda'):
    model.eval()
    running_loss = 0.0
    running_feature_loss = 0.0
    running_tv_loss = 0.0
    running_l1_loss = 0.0
    running_fft_loss = 0.0
    running_psnr = 0.0
    total_var_loss = 0.0
    final_loss = 0.0
    final_feature_loss = 0.0
    final_l1_loss = 0.0
    final_fft_loss = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
            image_data = data[0].to(device)
            label = data[1].to(device)

            outputs = model(image_data)
            loss = criterion(outputs, label).detach()
            feature_loss = vgg_loss(outputs, label).detach() * feature_loss_r
            total_var_loss = tv_loss(outputs).detach() * tv_loss_r
            l_loss = l1_loss(outputs, label).detach() * l1_loss_r
            f_loss = fft_loss(outputs, label).detach() * fft_loss_r
            total_loss = loss + total_var_loss + l_loss + f_loss
            # add loss of each item (total items in a batch = batch size)
            running_loss += loss.item()
            running_feature_loss += feature_loss.item()
            running_tv_loss += total_var_loss.item()
            running_l1_loss += l_loss.item()
            running_fft_loss += f_loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr
        outputs = outputs.cpu()
        #save_image(outputs, f"./output/val_sr{epoch}.png")
    final_loss = running_loss/len(dataloader.dataset)
    final_feature_loss = running_feature_loss/len(dataloader.dataset)
    #final_feature_loss = 0
    final_tv_loss = running_tv_loss/len(dataloader.dataset)
    final_l1_loss = running_l1_loss/len(dataloader.dataset)
    final_fft_loss = running_fft_loss/len(dataloader.dataset)
    final_psnr = running_psnr/int(len(dataloader.dataset)/dataloader.batch_size)
    return final_loss, final_psnr, final_feature_loss, final_tv_loss, final_l1_loss, final_fft_loss

def check_validation(model, dataloader, idx, device='cuda'):
    image_test = dataloader.dataset[idx][0]
    image_gt = dataloader.dataset[idx][1]
    with torch.no_grad():
        img = image_test.to(device)
        img = img.unsqueeze(0)
        output = model(img)
    output = output.cpu().data.numpy()
    cmax, cmin = max(image_gt.flatten()), min(image_gt.flatten())

    image_test = np.squeeze(image_test.to('cpu').data.numpy())
    image_gt = np.squeeze(image_gt.to('cpu').data.numpy())
    output = np.squeeze(output)
    psnr_before_train = img_psnr(image_test, image_gt)
    psnr_after_train = img_psnr(output, image_gt)

    plt.figure()
    plt.subplot(131);plt.imshow(image_test, clim=[cmin, cmax]);plt.title('initial')
    plt.subplot(132);plt.imshow(image_gt, clim=[cmin, cmax]);plt.title('ground truth')
    plt.subplot(133);plt.imshow(output, clim=[cmin, cmax]);plt.title('recover')
    plt.title(f'psnr: {psnr_before_train:.3f} --> {psnr_after_train:.3f}')
    return output, image_test, image_gt


def check_image_fitting(model, image, device='cuda', plot_flag=0, clim=[0,1], ax='off', title='', figsize=(14, 8)):
    model = model.to(device)
    if ax == 'off':
        axes = 'off'
    else:
        axes = 'on'
    if len(image.shape) == 2:
        img = np.expand_dims(image, 0)
    else:
        img = image.copy()
    with torch.no_grad():
        img = torch.tensor(img, dtype=torch.float).to(device)
        img = img.unsqueeze(0)
        img_fit = model(img)
    img_fit = img_fit.cpu().data.numpy()
    if plot_flag:
        plt.figure(figsize=figsize)
        if clim is None:
            clim = [np.min(image), np.max(image)]
        plt.subplot(121);plt.imshow(image, clim=clim);plt.axis(axes);plt.title('raw image')
        plt.subplot(122);plt.imshow(np.squeeze(img_fit), clim=clim);plt.axis(axes);plt.title('recoverd image')
        plt.suptitle(title)
    return np.squeeze(img_fit)



def img_psnr(img1, img2, d_range=None):
    from skimage.metrics import peak_signal_noise_ratio as psnr
    if d_range is None:
        data_range = max(np.max(img1), np.max(img2))
    else:
        data_range = d_range
    return psnr(img1.astype(np.float32), img2.astype(np.float32), data_range=data_range)


def img_ssim(img_ref, img):
    from skimage.metrics import structural_similarity as ssim
    return ssim(img_ref, img, data_range=img.max()-img.min())


def avg_smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def sino_smooth(sino, box_pts=3):
    sino_m = np.zeros(sino.shape) #(pix, angle)
    s = sino.shape
    for i in range(s[1]):
        sino_m[:, i] = avg_smooth(sino[:, i], box_pts)
    return sino_m



def rm_abnormal(img):
    img_c = img
    img_c[np.isnan(img_c)] = 0
    img_c[np.isinf(img_c)] = 0
    return img_c

def circ_mask(img, axis, ratio=1, val=0):
    im = np.float32(img)
    s = im.shape
    if len(s) == 2:
        m = _get_mask(s[0], s[1], ratio)
        m_out = (1 - m) * val
        im_m = np.array(m, dtype=np.int16) * im + m_out
    else:
        im = im.swapaxes(0, axis)
        dx, dy, dz = im.shape
        m = _get_mask(dy, dz, ratio)
        m_out = (1 - m) * val
        im_m = np.array(m, dtype=np.int16) * im + m_out
        im_m = im_m.swapaxes(0, axis)
    return im_m


def _get_mask(dx, dy, ratio):
    rad1 = dx / 2.
    rad2 = dy / 2.
    if dx > dy:
        r2 = rad1 * rad1
    else:
        r2 = rad2 * rad2
    y, x = np.ogrid[0.5 - rad1:0.5 + rad1, 0.5 - rad2:0.5 + rad2]
    return x * x + y * y < ratio * ratio * r2




def script_tranning():
    device = 'cuda:0'
    vgg19 = torchvision.models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    vgg19.to(device).eval()

    # load dataset
    fn1 = './tomo_blur' # directory for blurred images
    fn2 = './tomo_gt'  # directory for groundtruth images
    dataset = tomoDataset(fn1, fn2, 10000)
    n = len(dataset)
    batch_size = 8
    split_ratio = 0.8
    n_train = int(split_ratio * n)
    n_valid = n - n_train

    train_ds, valid_ds = torch.utils.data.random_split(dataset, (n_train, n_valid))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    len(train_loader.dataset)

    # model

    epochs = 300
    lr = 0.0002 # 0.001
    #device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("computration device: ", device)

    #in_nc, out_nc, nf, nb, gc
    model = RRDBNet(1, 1, 16, 4, 32).to(device)
    # load pre-trained model 
    model_path = './pre_trained_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # train the model
    train_loss, val_loss, train_tv_loss, train_l1_loss = [], [], [], []
    train_psnr, val_psnr = [], []
    best_psnr = 0
    start = time.time()
    for epoch in range(epochs+1):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_psnr, train_epoch_feature_loss, train_epoch_tv_loss, train_epoch_l1_loss, train_f_loss = train(model, train_loader, 1, 0, 1, 1, device) #for tomo, set 0.005
        val_epoch_loss, val_epoch_psnr, val_epoch_feature_loss, val_epoch_tv_loss, val_epoch_l1_loss, val_f_loss = validate(model, val_loader, epoch, 1, 0, 1, 1, device)# for sino, set 500
        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")
        print(f"train_loss = {train_epoch_loss:.3e}, feature_loss = {train_epoch_feature_loss:.3e}, tv_loss = {train_epoch_tv_loss:.3e}, l1_loss={train_epoch_l1_loss:.3e}, fft_loss={train_f_loss:.3e}")
        print(f"val_loss = {val_epoch_loss:.3e}, feature_loss = {val_epoch_feature_loss:.3e}, tv_loss = {val_epoch_tv_loss:.3e}, l1_loss={val_epoch_l1_loss:.3e}, fft_loss={val_f_loss:.3e}")
        print('\n\n')
        train_loss.append(train_epoch_loss+train_epoch_feature_loss)
        train_psnr.append(train_epoch_psnr)
        if train_psnr[-1] > best_psnr:
            best_psnr = train_psnr[-1]
            torch.save(model.state_dict(),'./best_model.pth')
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
        train_tv_loss.append(train_epoch_tv_loss)
    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes")

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
