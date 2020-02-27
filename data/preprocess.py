import os

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def save_to_h5(path, raw, label):
    with h5py.File(path, 'w') as f:
        f['raw'] = raw
        f['label'] = label


def read_sub_type(root, image_type='HGG'):
    filenames = os.listdir(os.path.join(root, image_type))

    raw_tails = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
    label_tail = '_seg.nii.gz'

    for filename in filenames:
        images = np.stack([np.array(nib_load(os.path.join(root, image_type, filename, filename + tail)), dtype='float32', order='C')
             for tail in raw_tails], -1)

        images = np.array(images)
        images = process_f32(images)  # C x H x W x D
        # TODO: transpose images???
        images = images.transpose(0, 3, 1, 2)  # C x D x H x W

        label = np.array(nib_load(os.path.join(root, image_type, filename, filename + label_tail)),
                         dtype='uint8', order='C')
        label[label == 4] = 3

        folder = os.path.join(root, 'h5', 'all')
        train_folder = os.path.join(root, 'h5', 'train')
        val_folder = os.path.join(root, 'h5', 'val')
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)

        print('--------------')
        print('file name:', filename)
        print('images:', images.mean(), images.min(), images.max())
        print('label:', label.mean(), label.min(), label.max())
        # print(raw.shape, label.shape)
        # save_to_h5(os.path.join(folder, filename + '.h5'), raw, label)


def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def process_f32(images):
    """ Set all Voxels that are outside of the brain mask to 0"""
    mask = images.sum(-1) > 0

    for k in range(4):
        x = images[..., k]  #
        y = x[mask]  #

        lower = np.percentile(y, 0.2)  # 算分位数
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        print('y: ', 'mean:', y.mean(), 'std:', y.std(), 'lower:', lower, 'upper:', upper)
        y = x[mask]

        x -= y.mean()
        x /= y.std()

        images[..., k] = x

    return images


def read_data(root):
    for image_type in ['HGG', 'LGG']:
        read_sub_type(root, image_type)

def split_data(root, train_file, val_file):
    with open(train_file, 'r') as f:
        train_list = f.readlines()
        temp = [name[:-1].split('/')[1] for name in train_list[:-1]]
        train_list = temp + [train_list[-1].split('/')[1]]

    with open(val_file, 'r') as f:
        val_list = f.readlines()
        temp = [name[:-1].split('/')[1] for name in val_list[:-1]]
        val_list = temp + [val_list[-1].split('/')[1]]

    # move file from all folder to train and val_folder
    all_folder = os.path.join(root, 'all')
    train_folder = os.path.join(root, 'train')
    val_folder = os.path.join(root, 'val')
    # 1: move from train/val to all if there is existed
    for name in os.listdir(train_folder):
        os.rename(os.path.join(train_folder, name), os.path.join(all_folder, name))
    for name in os.listdir(val_folder):
        os.rename(os.path.join(val_folder, name), os.path.join(all_folder, name))

    # 2: move from all to train and val
    for name in train_list:
        name = name + '.h5'
        os.rename(os.path.join(all_folder, name), os.path.join(train_folder, name))
    for name in val_list:
        name = name + '.h5'
        os.rename(os.path.join(all_folder, name), os.path.join(val_folder, name))


def main():
    root = '../../data/2018/MICCAI_BraTS_2018_Data_Training'
    read_data(root)
    split_data(os.path.join(root, 'h5'), os.path.join(root, 'train_0.txt'), os.path.join(root, 'valid_0.txt'))


if __name__ == '__main__':
    main()
