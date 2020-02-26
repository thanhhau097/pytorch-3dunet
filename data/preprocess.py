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
            raw.append(img.get_data().astype(np.uint8)) # TODO: convert to int8

        raw = np.array(raw)
        label = nib.load(os.path.join(root, image_type, filename, filename + label_tail)).get_data().astype(np.uint8)

        folder = os.path.join(root, 'h5', image_type)
        if not os.path.exists(folder):
            os.makedirs(folder)

        save_to_h5(os.path.join(folder, filename + '.h5'), raw, label)


def read_data(root):
    for image_type in ['HGG', 'LGG']:
        read_sub_type(root, image_type)


def list_files(root, kind='train', output='all.txt'):
    files = []
    if kind == 'train':
        subs = ['HGG', 'LGG']
        for sub in subs:
            files += [os.path.join(sub, file) for file in os.listdir(os.path.join(root, sub))]
    else:
        files = [file for file in os.listdir(root)]

    with open(os.path.join(root, output), 'w') as f:
        for file in files:
            f.write(file)
            f.write('\n')

def main():
    root = '2018/MICCAI_BraTS_2018_Data_Training'
    list_files('2018/MICCAI_BraTS_2018_Data_Training')
    list_files('2018/MICCAI_BraTS_2018_Data_Validation', kind='val', output='val.txt')



if __name__ == '__main__':
    main()
