import random
import torch
import logging
import sys
sys.path.append('../')
# from new_dataloader import DataGenerator
from dataloader_117 import DataGenerator

import torch.optim as optim
from new_vit_hpe_1_12 import HPEViT
from new_vit import ViTPose
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
# from config.utils import create_logger, get_model_summary, get_optimizer
import os
import argparse
from torch.utils.data import DataLoader
import numpy as np
from tools.utils import init_dir, IOStream, decode_batch_sa_simdr, accuracy, KLDiscretLoss
from tools.geometry_function import get_pred_3d_batch, cal_2D_mpjpe, cal_3D_mpjpe

# current_path = os.getcwd()
# print("current_path:", current_path)


# 日志配置
# logging.basicConfig(filename='/home/ynn/Event-Based_HPE/our_DHP19/ourmethod_pytorch/vit_training_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
logging.basicConfig(filename='/mnt/DHP19_our/ourmethod_mt/vit_training_resize118_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def mse2D(y_true, y_pred):
    mean_over_ch = torch.mean((y_pred - y_true) ** 2, dim=-1)
    mean_over_w = torch.mean(mean_over_ch, dim=-1)
    mean_over_h = torch.mean(mean_over_w, dim=-1)
    return torch.mean(mean_over_h)
    # mean_over_ch = torch.sum((y_pred - y_true) ** 2, dim=-1)
    # mean_over_w = torch.sum(mean_over_ch, dim=-1)
    # mean_over_h = torch.sum(mean_over_w, dim=-1)
    # return torch.sum(mean_over_h)


def predict_CNN_extract_skeleton_2d_batch(outputs, labels, batch_size=4):
    pred = outputs.detach().cpu().numpy()  # shape torch.Size([4, 13, 260, 344]),
    labels = labels.detach().cpu().numpy()

    sample_mpjpe = []
    y_2d_float_batch = []
    p_coords_max_batch = []

    for b in range(batch_size):
        y_2d = np.zeros((13, 2))
        for j_idx in range(13):
            label_j_map = labels[b, j_idx, :, :]
            l_coords_max_tmp = np.argwhere(label_j_map == np.max(label_j_map))
            y_2d[j_idx] = l_coords_max_tmp[0]

        p_coords_max = np.zeros((13, 2))
        for j_idx in range(y_2d.shape[0]):
            pred_j_map = pred[b, :, :, j_idx]
            p_coords_max_tmp = np.argwhere(pred_j_map == np.max(pred_j_map))
            p_coords_max[j_idx] = p_coords_max_tmp[0]

        y_2d_float = y_2d.astype(np.float64)
        dist_2d = np.linalg.norm((y_2d_float - p_coords_max), axis=-1)
        mpjpe = np.nanmean(dist_2d)
        sample_mpjpe.append(mpjpe)

        y_2d_float_batch.append(y_2d_float)
        p_coords_max_batch.append(p_coords_max)

    mpjpe_ave = sum(sample_mpjpe) / len(sample_mpjpe)
    return mpjpe_ave


class JointsMSELoss(torch.nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * torch.mean(
                    target_weight[:, idx] *
                    (heatmap_pred - heatmap_gt) ** 2
                )
            else:
                loss += 0.5 * torch.mean(
                    (heatmap_pred - heatmap_gt) ** 2
                )

        return loss / num_joints



# # 定义学习率调度器
# def scheduler_func(epoch):
#     # if epoch < 1000:  # when epoch=20000
#     # if epoch < 100:  # when epoch=200
#     # if epoch < 1000:  # when epoch=2000
#     if epoch < 1000:  # when epoch=5000
#         return 0.01
#     # elif epoch < 5000:  # when epoch=20000
#     # elif epoch < 150:  # when epoch=200
#     # elif epoch < 1500:  # when epoch=2000
#     elif epoch < 2000:  # when epoch=5000
#         return 0.001
#     elif epoch < 3500:  # when epoch=5000
#         return 0.0005
#     else:
#         return 0.0001


def scheduler_func(epoch):
    if epoch < 10:
        return 0.01
    elif epoch < 15:
        return 0.001
    else:
        return 0.0001


def train(args):
    # num_of_files = 96
    # num_of_val_files = 24
    # list_IDs = random.sample(range(num_of_files), num_of_files)
    #
    # train_set = list_IDs[:-num_of_val_files]  # list_IDs =
    # validation_set = list_IDs[-num_of_val_files:]  # len(validation_set) = 24,在所有文件（96个）中随机选出24个作为测试文件，下列为对应测试文件的文件的ID
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
        patch_size=(9, 16),  # 共有260 / 13 = 20， 344 * 8 = 43个patch
        num_joints=13,
        dim=512,
        depth=6,  # encode堆叠的个数，就是多少个encode
        heads=6,  # 多头注意分为多少个头，也就是需要几个注意力
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
    # model = ViTPose(
    #         image_height=720,
    #         image_width=1280,
    #         patch_size=(16, 16),  # 共有224 / 16 = 14， 14 * 14 = 96个patch
    #         num_keypoints=13,
    #         dim=768,
    #         depth=1,  # encode堆叠的个数，就是多少个encode
    #         heads=16,  # 多头注意分为多少个头，也就是需要几个注意力
    #         mlp_dim=1024,
    #         dropout=0.1,
    #         emb_dropout=0.1
    # )
    # model = nn.DataParallel(model)
    device = torch.device("cuda:{}".format(args.cuda_num) if args.cuda else "cpu")
    model.to(device)
    
    target_weights = torch.ones(1, 13, 1).to(device)
    # criterion = JointsMSELoss(use_target_weight=True)
    # criterion = torch.nn.BCELoss()
    # criterion = mse2D  # 损失函数
    criterion = KLDiscretLoss()
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    # optimizer = get_optimizer(cfg, model)
    # 将调度器应用到优化器上
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: scheduler_func(epoch))

    # 模型训练
    num_epochs = args.epochs
    best_miou = 0.0  # 初始化最佳mIoU为0
    num_classes = 13  # 假设有13个类别

    # # 加载检查点文件以恢复训练
    # checkpoint_path = '/home/ynn/Event-Based_HPE/AFFormer/tools/work_dirs/Adpt_token_attn-carla/203_model.pth'  # 检查点文件路径
    # if os.path.isfile(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     # print('checkpoint.keys', checkpoint.keys())
    #     # model.load_state_dict(checkpoint['model_state_dict'])
    #     model.load_state_dict(checkpoint)
    #     if 'optimizer_state_dict' in checkpoint:
    #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     else:
    #         print("No optimizer state found in checkpoint!")
    #     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     # resume_epoch = checkpoint['epoch']
    #     if 'epoch' in checkpoint:
    #         start_epoch = checkpoint['epoch']
    #     else:
    #         start_epoch = 204
    #     # best_miou = checkpoint['best_miou']
    #     print(f"Successfully loaded checkpoint '{checkpoint_path}' at epoch {resume_epoch}")
    # else:
    #     print(f"No checkpoint found at '{checkpoint_path}', starting from scratch")
    
    min_loss = 1e6
    for epoch in range(1, num_epochs + 1):
        optimizer.step()
        running_loss = 0.0
        val_loss_sum = 0.0
        # min_loss = 1e6
        model.train()  # 设置为训练模式
        # print("train--------")
        for i, (data, x, y, weight) in enumerate(train_loader, 0):
            optimizer.zero_grad()  # 梯度清零
            inputs = data.to(device)   # torch.Size([4, 1, 260, 344])
            x, y, weight = x.to(device), y.to(device), weight.to(device)  # torch.Size([4, 13, 260, 344])
            # print("inputs : ", inputs.shape)
            # return
            # 前向传播
            outputs_x, outputs_y = model(inputs)  # outputs.shape torch.Size([4, 13, 260, 344])
            # print("outputs : ", outputs)
            # loss = criterion(outputs, labels, target_weights)
            loss = criterion(outputs_x, outputs_y, x, y, weight)
            # print("{} epoch train loss : {}".format(epoch, loss))

            # 统计损失、反向传播和优化
            running_loss += loss.item()
            loss.backward()
            scheduler.step()

        # 每个 epoch 结束后清理内存
        torch.cuda.empty_cache()
        train_loss_ave = running_loss / i

        # 记录训练损失到日志
        logging.info(f'Epoch {epoch}: Train Loss: {train_loss_ave}')
        print("{} epoch train loss : {}".format(epoch, train_loss_ave))

        if epoch % 1 == 0:  # when epoch=200
        # if epoch % 50 == 0:  # when epoch=200
        # if epoch % 500 == 0:  # when epoch=20000
            # 在每个epoch后对测试集进行评估
            model.eval()
            with torch.no_grad():
                mpjpe_loss = 0
                for i, (val_data, val_x, val_y, val_weight) in enumerate(test_loader, 0):
                    val_inputs = val_data.to(device)
                    val_x, val_y, val_weight = val_x.to(device), val_y.to(device), val_weight.to(device)

                    val_output_x, val_output_y = model(val_inputs)
                    # val_loss = criterion(outputs, labels)
                    # batch_mpjpe_loss = predict_CNN_extract_skeleton_2d_batch(outputs, labels, args.valid_batch_size)
                    decode_batch_label = decode_batch_sa_simdr(val_x, val_y)
                    decode_batch_pred = decode_batch_sa_simdr(val_output_x, val_output_y)

                    pred = np.zeros((args.valid_batch_size, 13, 2))
                    pred[:, :, 1] = decode_batch_pred[:, :, 0]  # exchange x,y order
                    pred[:, :, 0] = decode_batch_pred[:, :, 1]

                    batch_mpjpe_loss = cal_2D_mpjpe(decode_batch_label, val_weight.squeeze(dim=2).cpu(), decode_batch_pred)
                    
                    # 统计损失
                    # running_loss += val_loss.item()
                    mpjpe_loss += batch_mpjpe_loss

                # val_loss_ave = running_loss / i
                mpjpe_loss_ave = mpjpe_loss / i

                logging.info(f'Epoch {epoch}: Test mpjpe  Loss: {mpjpe_loss_ave}')
                print("{} epoch mpjpe loss : {}".format(epoch, mpjpe_loss_ave))

                # 检查是否为最佳模型并保存
                if mpjpe_loss_ave < min_loss:
                    min_loss = mpjpe_loss_ave
                    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, args.save_model + '/model_resize_p9_12h_d1_512_118.pth')
                    print(f'Epoch {epoch}: New best model saved with loss: {min_loss}')
                    # 记录到日志

            # # 可选: 每200个epoch输出一次
            # if epoch % 200 == 0:
            #     print(
            #         f'Epoch {epoch}: Train Loss: {train_loss_ave}')

        # 最终的测试结果
        # logging.info(f'Final Test Results - mIoU: {best_miou}, MPA: {avg_mpa}')


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
    parser.add_argument('--num_joints', type=int, default=13,
                        help='number of joints')
    parser.add_argument('--save_model', type=str, default='/mnt/DHP19_our/ourmethod_mt/save_model',
                        help='the path where the model is saved')
    parser.add_argument('--cuda_num', type=int, default=0, metavar='N',
                        help='cuda device number')
    parser.add_argument('--sigma', type=int, default=4, metavar='N',
                        help='sigma value')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    train(args)

    print('******** Finish HPE VIT ********')
