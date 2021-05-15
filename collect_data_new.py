"""Giving folders of patch images, make DataFrame """

import pandas as pd
import pickle
import cv2 as cv
import os


def re_index_img_stack():
    cnt_id = 0
    resize_to: tuple = None
    save_dir = 'F:/Dataset/patch_dataset/kitti_raw_re_index'
    orig_stack_dir = 'F:/Dataset/patch_dataset/kitti_raw'
    save_ext = '.jpg'
    for root, dirs, files in os.walk(orig_stack_dir):
        for file in files:
            if file.endswith('.jpg'):
                filepath = os.path.join(root, file).replace('\\','/')
                print(filepath)
                frame = cv.imread(filepath)
                if resize_to is not None:
                    frame = cv.resize(frame, resize_to)
                savename = '{:06}.{}'.format(cnt_id, save_ext)
                cnt_id += 1
                if save_dir is None:
                    save_dir = orig_stack_dir + '_re_index'
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                savepath = os.path.join(save_dir, savename)
                cv.imwrite(savepath, frame)


def make_onehot(lab_numm, total_num):
    lab_onehot = [0] * total_num
    lab_onehot[lab_numm] = 1
    return lab_onehot


def create_df():
    filename_list = []
    class_label_list = []
    class_one_hot_list = []
    save_filename = 'vehicle.pkl'
    label_names = {0: 'vehicle'}
    lab_num = 0     # for now only "vehicle" class
    lab_onehot = make_onehot(lab_num, 3)
    dataset_root = 'F:/Dataset/patch_dataset'
    save_path = os.path.join(dataset_root, save_filename)
    print('-> Creating DataFrame, saved in {}'.format(dataset_root))
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.jpg'):
                filepath = os.path.join(root, file)
                filename_list.append(filepath)
                class_label_list.append(label_names[lab_num])
                class_one_hot_list.append(lab_onehot)

    # save res to pkl file
    print("-> Saving as pickle file")
    cols = ['filename', 'class_label', 'class_one_hot']
    pk_out = pd.DataFrame({'filename': filename_list,
                           'class_label': class_label_list,
                           'class_one_hot': class_one_hot_list})
    pk_out = pk_out[cols]
    print(pk_out)
    pd.to_pickle(pk_out, save_path)
    print("pickle saved")


def check_df():
    df = pd.read_pickle("")
    for ind, row in df.iterrows():
        old_file, class_label, class_one_hot = row['filename'], row['class_label'], row['class_one_hot']


if __name__ == '__main__':
    # check_df()
    # re_index_img_stack()
    create_df()
