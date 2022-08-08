import os
import random
import numpy as np
from tqdm import tqdm
from glob import glob

import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import SegModel
from dataset import SegDataset
from metrics import mean_IOU, pixel_acc
from data_preprocessing import TRANSFORMS
from get_model import get_model

from net_params import (
    DATASET_NAME,
    EPOCHS,
    BATCH_SIZE,
    TRAIN_VAL_SPLIT,
    LR,
    GAMMA,
    MODEL_CHOICE
)


def get_train_val_split(images_path, masks_path, ratio):
    random.seed(23)
    img_list = glob(images_path + '/*')
    mask_list = glob(masks_path + '/*')

    map_idx_position = list(zip(img_list, mask_list))
    random.shuffle(map_idx_position)
    images_shuffled, masks_shuffled = zip(*map_idx_position)

    images_shuffled = list(images_shuffled)
    masks_shuffled = list(masks_shuffled)

    train_images = images_shuffled[0: int(ratio * len(img_list))]
    train_masks = masks_shuffled[0: int(ratio * len(img_list))]
    val_images = images_shuffled[int(ratio * len(img_list)):]
    val_masks = masks_shuffled[int(ratio * len(img_list)):]

    # save these lists to a file so i can know which images are used in train/val for tests later
    fname = os.path.join(DATASET_NAME, 'train_val_distro.json')
    with open(fname, 'w') as f:
        json.dump({
            'train': train_images, 'val': val_images
        }, f)

    t_ds = SegDataset(images_list=train_images,
                      masks_list=train_masks,
                      tfms_img=TRANSFORMS['train']['image'],
                      tfms_geo=TRANSFORMS['train']['geo'])
    v_ds = SegDataset(images_list=val_images,
                      masks_list=val_masks,
                      tfms_img=TRANSFORMS['val']['image'],
                      tfms_geo=None)

    return t_ds, v_ds


im_path = os.path.join(DATASET_NAME, 'images')
msk_path = os.path.join(DATASET_NAME, 'masks')

net = get_model()
# USE_CKPT = None
USE_CKPT = os.path.join('checkpoints', 'fromscratch_celeba_4000px_deeplab101_2_0.0001_0.95_30.pt')
fine_tune = 'finetune' if USE_CKPT else 'fromscratch'
print('checkpoint: {}'.format(USE_CKPT))

train_ds, val_ds = get_train_val_split(im_path, msk_path, TRAIN_VAL_SPLIT)
print('Dataset  size: train: {} val: {}'.format(len(train_ds), len(val_ds)))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = SegModel(net=net)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=GAMMA)


# implement a stop training func if you want to automatically stop when it begins overfitting
def training_loop(n_epochs, optimizer, lr_scheduler, model, loss_fn, train_loader, val_loader, last_ckpt_pth=None):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    tr_loss_arr = []
    val_loss_arr = []
    mean_iou_train = []
    pixel_acc_train = []
    mean_iou_val = []
    pixel_acc_val = []
    prev_epoch = 0

    if last_ckpt_pth is not None:
        checkpoint = torch.load(last_ckpt_pth)
        prev_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
                    tr_loss_arr = checkpoint['Training Loss']
        val_loss_arr = checkpoint['Validation Loss']
        mean_iou_train = checkpoint['MeanIOU train']
        pixel_acc_train = checkpoint['PixelAcc train']
        mean_iou_val = checkpoint['MeanIOU test']
        pixel_acc_val = checkpoint['PixelAcc test']
        print("loaded model, ", checkpoint['description'], "at epoch", prev_epoch)
        model.to(device)

    model.to(device)

    for epoch in range(0, n_epochs):

        pbar = tqdm(train_loader, total=len(train_loader))
        for X, y in pbar:
            torch.cuda.empty_cache()
            model.train()
            X = X.to(device).float()
            y = y.to(device).float()
            ypred = model(X)
            loss = loss_fn(ypred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss_arr.append(loss.item())
            mean_iou_train.append(mean_IOU(y, ypred))
            pixel_acc_train.append(pixel_acc(y, ypred))
            pbar.set_postfix({'Epoch': epoch + 1 + prev_epoch,
                              'Training Loss': np.mean(tr_loss_arr),
                              'Mean IOU': np.mean(mean_iou_train),
                              'Pixel Acc': np.mean(pixel_acc_train)
                              })

        with torch.no_grad():
            pbar = tqdm(val_loader, total=len(val_loader))
            for X, y in pbar:
                torch.cuda.empty_cache()
                X = X.to(device).float()
                y = y.to(device).float()
                model.eval()
                ypred = model(X)

                val_loss_arr.append(loss_fn(ypred, y).item())
                pixel_acc_val.append(pixel_acc(y, ypred))
                mean_iou_val.append(mean_IOU(y, ypred))

                pbar.set_postfix({'Epoch': epoch + 1 + prev_epoch,
                                  'Validation Loss': np.mean(val_loss_arr),
                                  'Mean IOU': np.mean(mean_iou_val),
                                  'Pixel Acc': np.mean(pixel_acc_val)
                                  })

        checkpoint = {
            'epoch': epoch + 1 + prev_epoch,
            'description': "{}".format(MODEL_CHOICE),
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Training Loss': tr_loss_arr,
            'Validation Loss': val_loss_arr,
            'MeanIOU train': mean_iou_train,
            'PixelAcc train': pixel_acc_train,
            'MeanIOU test': mean_iou_val,
            'PixelAcc test': pixel_acc_val
        }
        if not (epoch + 1) % 10:
            torch.save(checkpoint, 'checkpoints/{}_{}_{}_{}_{}_{}_{}.pt'.format(
                fine_tune,
                DATASET_NAME.split('dataset_')[-1],
                MODEL_CHOICE,
                BATCH_SIZE,
                LR,
                GAMMA,
                str(epoch + 1 + prev_epoch)))

        lr_scheduler.step()

    save_results = 'dataset: {}-{}\tmodel: {}\tepochs: {}\t' \
                   'batch size: {}\tlearning rate: {}\tgamma : {}\t' \
                   'train loss: {:.6f}\t val loss: {:.6f}\t' \
                   'mean IOU (train): {:.6f}\tpixel acc (train): {:.6f}\t' \
                   'mean IOU (val): {:.6f}\tpixel acc (val): {:.6f}\t'.format(
        DATASET_NAME, len(train_ds), MODEL_CHOICE, EPOCHS,
        BATCH_SIZE, LR, GAMMA,
        np.mean(tr_loss_arr), np.mean(val_loss_arr),
        np.mean(mean_iou_train), np.mean(pixel_acc_train),
        np.mean(mean_iou_val), np.mean(pixel_acc_val)
    )

    with open('results.txt', 'a') as f:
        f.write(save_results)
        f.write('\n')
    return tr_loss_arr, val_loss_arr, mean_iou_train, pixel_acc_train, mean_iou_val, pixel_acc_val


if __name__ == '__main__':
    ret_vals = training_loop(EPOCHS, optimizer, lr_scheduler, model, loss_fn, train_loader, val_loader,
                             last_ckpt_pth=USE_CKPT)

