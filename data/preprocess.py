import os

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def save_to_h5(path, raw, label):
    with h5py.File(path, 'w') as f:
        f['raw'] = raw
        f['label'] = label


def read_sub_type(root, image_type='HGG'):
    filenames = os.listdir(os.path.join(root, image_type))

    raw_tails = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz', '_t2.nii.gz']
    label_tail = '_seg.nii.gz'

    for filename in tqdm(filenames):
        raw = []
        for tail in raw_tails:
            path = os.path.join(root, image_type, filename, filename + tail)
            img = nib.load(path)
            raw.append(img.get_fdata().astype(np.uint8)) # TODO: convert to int8

        raw = np.array(raw)
        label = nib.load(os.path.join(root, image_type, filename, filename + label_tail)).get_fdata().astype(np.uint8)

        folder = os.path.join(root, 'h5', image_type)
        if not os.path.exists(folder):
            os.makedirs(folder)

        save_to_h5(os.path.join(folder, filename + '.h5'), raw, label)


def read_data(root):
    for image_type in ['HGG', 'LGG']:
        read_sub_type(root, image_type)


def main():
    root = '2018/MICCAI_BraTS_2018_Data_Training'
    read_data(root)


if __name__ == '__main__':
    main()
