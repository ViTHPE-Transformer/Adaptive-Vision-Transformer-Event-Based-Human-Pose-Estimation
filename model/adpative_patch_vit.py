import random
import torch
import logging
import sys
sys.path.append('../')
# from new_dataloader import DataGenerator
from dataloader import DataGenerator

import torch.optim as optim
from vit import HPEViT
from adpative_patch_vit import ViTPose
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
# from config.utils import create_logger, get_model_summary, get_optimizer
import os
import argparse
from torch.utils.data import DataLoader
import numpy as np
from tools.utils import init_dir, IOStream, decode_batch_sa_simdr, accuracy, KLDiscretLoss
from tools.geometry_function import get_pred_3d_batch, cal_2D_mpjpe, cal_3D_mpjpe


# 日志配置
# logging.basicConfig(filename='/home/ynn/Event-Based_HPE/our_DHP19/ourmethod_pytorch/vit_training_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logging.basicConfig(filename='/mnt/DHP19_our/ourmethod_mt/vit_training_resize118_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def scheduler_func(epoch):
    if epoch < 10:
        return 0.01
    elif epoch < 15:
        return 0.001
    else:
        return 0.0001


def train(args):
    root_data_train = "/mnt/DHP19_our/train/"
    root_data_test = "/mnt/DHP19_our/test/"
    train_dataset = DataGenerator(root_data_dir=root_data_train + "data",
                                  root_label_dir=root_data_train + "label",
                                  root_dict_dir=root_data_train + "Frame_dict.npy",
                                  size_h=args.sensor_sizeH,
                                  size_w=args.sensor_sizeW,
                                  joints=args.num_joints)
    test_dataset = DataGenerator(root_data_dir=root_data_test + "data",
                                 root_label_dir=root_data_test + "label",
                                 root_dict_dir=root_data_test + "Frame_dict.npy",
                                 size_h=args.sensor_sizeH,
                                 size_w=args.sensor_sizeW,
                                 joints=args.num_joints)
    train_loader = DataLoader(train_dataset,
                              num_workers=8,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_dataset,
                             num_workers=8,
                             batch_size=args.valid_batch_size,
                             shuffle=False,
                             drop_last=False)

    model = HPEViT(
        image_size=(180, 320),
        patch_size=(9, 16),
        num_joints=17,
        dim=512,
        depth=6,
        heads=6,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
    # model = ViTPose(
    #         image_height=720,
    #         image_width=1280,
    #         patch_size=(16, 16),
    #         num_keypoints=17,
    #         dim=768,
    #         depth=1,
    #         heads=16,
    #         mlp_dim=1024,
    #         dropout=0.1,
    #         emb_dropout=0.1
    # )
    # model = nn.DataParallel(model)
    device = torch.device("cuda:{}".format(args.cuda_num) if args.cuda else "cpu")
    model.to(device)

    criterion = KLDiscretLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: scheduler_func(epoch))

    num_epochs = args.epochs

    min_loss = 1e6
    for epoch in range(1, num_epochs + 1):
        optimizer.step()
        running_loss = 0.0
        model.train()

        for i, (data, x, y, weight) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            inputs = data.to(device)
            x, y, weight = x.to(device), y.to(device), weight.to(device)

            outputs_x, outputs_y = model(inputs)

            loss = criterion(outputs_x, outputs_y, x, y, weight)

            running_loss += loss.item()
            loss.backward()
            scheduler.step()

        torch.cuda.empty_cache()
        train_loss_ave = running_loss / i

        logging.info(f'Epoch {epoch}: Train Loss: {train_loss_ave}')
        print("{} epoch train loss : {}".format(epoch, train_loss_ave))

        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                mpjpe_loss = 0
                for i, (val_data, val_x, val_y, val_weight) in enumerate(test_loader, 0):
                    val_inputs = val_data.to(device)
                    val_x, val_y, val_weight = val_x.to(device), val_y.to(device), val_weight.to(device)

                    val_output_x, val_output_y = model(val_inputs)

                    decode_batch_label = decode_batch_sa_simdr(val_x, val_y)
                    decode_batch_pred = decode_batch_sa_simdr(val_output_x, val_output_y)

                    pred = np.zeros((args.valid_batch_size, 13, 2))
                    pred[:, :, 1] = decode_batch_pred[:, :, 0]
                    pred[:, :, 0] = decode_batch_pred[:, :, 1]

                    batch_mpjpe_loss = cal_2D_mpjpe(decode_batch_label, val_weight.squeeze(dim=2).cpu(), decode_batch_pred)

                    mpjpe_loss += batch_mpjpe_loss

                mpjpe_loss_ave = mpjpe_loss / i

                logging.info(f'Epoch {epoch}: Test mpjpe  Loss: {mpjpe_loss_ave}')
                print("{} epoch mpjpe loss : {}".format(epoch, mpjpe_loss_ave))

                if mpjpe_loss_ave < min_loss:
                    min_loss = mpjpe_loss_ave
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, args.save_model + '/model_resize_p9_12h_d1_512_118.pth')
                    print(f'Epoch {epoch}: New best model saved with loss: {min_loss}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HPE VIT')
    parser.add_argument('--train_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of train batch)')
    parser.add_argument('--valid_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of valid batch)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--sensor_sizeH', type=int, default=720,
                        help='sensor_sizeH')
    parser.add_argument('--sensor_sizeW', type=int, default=1280,
                        help='sensor_sizeW')
    parser.add_argument('--num_joints', type=int, default=17,
                        help='number of joints')
    parser.add_argument('--save_model', type=str, default='/mnt/DHP19_our/ourmethod_mt/save_model',
                        help='the path where the model is saved')
    parser.add_argument('--cuda_num', type=int, default=0, metavar='N',
                        help='cuda device number')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train(args)

    print('******** Finish HPE VIT ********')
