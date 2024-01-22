import tensorflow as tf
import torch
import numpy as np
import random
import os
import cv2


def adjust_target_weight(joint, target_weight, tmp_size, sx=int(1280 / 4), sy=int(720 / 4)):
    mu_x = joint[0]
    mu_y = joint[1]
    # Check that any part of the gaussian is in-bounds
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= sx or ul[1] >= sy or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        target_weight = 0

    return target_weight


def generate_sa_simdr(joints, joints_vis, sigma=8, sx=int(1280 / 4), sy=int(720 / 4), num_joints=13):
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target_x = np.zeros((num_joints, int(sx)), dtype=np.float32)
    target_y = np.zeros((num_joints, int(sy)), dtype=np.float32)

    tmp_size = sigma * 3

    frame_size = np.array([sx, sy])
    frame_resize = np.array([sx, sy])
    feat_stride = frame_size / frame_resize

    for joint_id in range(num_joints):
        target_weight[joint_id] = \
            adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
        if target_weight[joint_id] == 0:
            continue

        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

        x = np.arange(0, int(sx), 1, np.float32)
        y = np.arange(0, int(sy), 1, np.float32)

        v = target_weight[joint_id]
        if v > 0.5:
            target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))
            target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))

            # norm to [0,1]
            target_x[joint_id] = (target_x[joint_id] - target_x[joint_id].min()) / (
                    target_x[joint_id].max() - target_x[joint_id].min())
            target_y[joint_id] = (target_y[joint_id] - target_y[joint_id].min()) / (
                    target_y[joint_id].max() - target_y[joint_id].min())

    return target_x, target_y, target_weight


def generate_label(u, v, mask, sigma=8, sx=int(1280 / 4), sy=int(720 / 4), num_joints=13):
    joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
    joints_3d_vis = np.zeros((num_joints, 3), dtype=np.float32)
    joints_3d[:, 0] = u
    joints_3d[:, 1] = v
    joints_3d_vis[:, 0] = mask
    joints_3d_vis[:, 1] = mask

    gt_x, gt_y, gt_joints_weight = generate_sa_simdr(joints_3d, joints_3d_vis, sigma, sx=sx, sy=sy)

    return gt_x, gt_y, gt_joints_weight


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, root_data_dir="", root_label_dir="", root_dict_dir="", size_w=1280, size_h=720, joints=13):
        self.root_data_dir = root_data_dir
        self.root_label_dir = root_label_dir
        self.root_dict_dir = root_dict_dir
        self.sizeW = size_w
        self.sizeH = size_h
        self.joints = joints
        self.dict = np.load(root_dict_dir, allow_pickle=True)

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):

        data, x, y, weight = self.__data_generation(index)

        # data, label = torch.tensor(data, dtype=torch.double), torch.tensor(label, dtype=torch.double)

        data, x, y, weight = torch.tensor(data, dtype=torch.float), torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float), torch.tensor(weight, dtype=torch.float)
        return data, x, y, weight

    def __data_generation(self, index):
        data = np.zeros((1, int(self.sizeH / 4), int(self.sizeW / 4)))
        # label = np.zeros((self.joints, self.sizeH, self.sizeW))

        data_file = np.load(os.path.join(self.root_data_dir, str(self.dict[index]) + ".npy"), allow_pickle=True).item()
        label_file = np.load(os.path.join(self.root_label_dir, str(self.dict[index]) + "_label.npy"), allow_pickle=True)

        for k, v in data_file.items():
            data[0][int(int(k.split(',')[1]) / 4)][int(int(k.split(',')[0])/4)] = v

        # u, v = label_file[:].astype(np.float32)
        u = label_file[:, 0] / 4
        v = label_file[:, 1] / 4
        mask = np.ones(13)
        x, y, weight = generate_label(u, v, mask)

        return data, x, y, weight





