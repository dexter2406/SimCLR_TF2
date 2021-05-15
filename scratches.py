import tensorflow as tf
from SimCLR import SimCLR
from DataGeneratorSimCLR import DataGeneratorSimCLR as DataGenerator
from SimCLR import SimCLR
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.utils import get_custom_objects
from sklearn.model_selection import train_test_split
from evaluate_features import get_features, linear_classifier, tSNE_vis
from swish import *
from SoftmaxCosineSim import SoftmaxCosineSim
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models.resnet18 import build_res18_enc
from models.wide_resnet_v1 import build_wide_resnet
from absl import app, flags
import os
import cv2 as cv
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string('split_filepath', 'F:/Dataset/patch_dataset/vehicle.pkl', 'where to store split info')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('height', 80, 'height of input')
flags.DEFINE_integer('width', 80, 'width of input')
flags.DEFINE_integer('num_epochs', 50, 'total number of epochs')
flags.DEFINE_list('feat_dims_ph', [128, 64], 'out dimension for projection head')
flags.DEFINE_string('weights_path', 'models/trashnet/SimCLR/SimCLR_05_15_22h_02.h5', 'trained weights')
flags.DEFINE_string('run_mode', 'test', 'choose from [train, test]')

# 'models/trashnet/SimCLR/SimCLR_05_15_19h_04.h5'
get_custom_objects().update({'Swish': Swish(swish)})
get_custom_objects().update({'SoftmaxCosineSim': SoftmaxCosineSim})


def main(_):
    opt = FLAGS
    if opt.run_mode == 'train':
        start_training(opt)
    elif opt.run_mode == 'test':
        field_test(opt)


def field_test(opt):
    params = {
        'input_shape': (80, 80, 3),
        'depth': 10,
        'width': 2,
        'drop_prob': 0.2,
        'out_units': 128
    }

    # extractor = build_wide_resnet(params, has_top=False)
    get_custom_objects().update({'Swish': Swish(swish)})
    get_custom_objects().update({'SoftmaxCosineSim': SoftmaxCosineSim})
    SimCLR = prepare_models(opt)
    model = get_proj_model(SimCLR)
    print('load weights from', opt.weights_path)
    model.load_weights(opt.weights_path, by_name=True, skip_mismatch=False)
    file_dir = 'F:/Dataset/patch_dataset/test_samples'
    frames = []
    for root, _, files in os.walk(file_dir):
        for file in files:
            if file.endswith('jpg'):
                print(file)
                filepath = os.path.join(root, file)
                frame = cv.imread(filepath)
                frames.append(frame)

    cos_sims = []
    num = len(frames)
    print(num, 'frames in total')
    for i in range(num):
        cos_sim = []
        for j in range(num):
            # plt.imshow(frames[i]), plt.show()
            # plt.imshow(frames[j]), plt.show()
            frame1 = tf.expand_dims(frames[i], 0)
            frame2 = tf.expand_dims(frames[j], 0)
            feat1 = model.predict(frame1)
            feat2 = model.predict(frame2)
            cos_sim.append(cosine_similarity(
                post_pool(feat1), post_pool(feat2)
            ))
        cos_sims.append(np.hstack(cos_sim))
    print("cosine sim mat:\n", np.vstack(cos_sims))


def prepare_models(opt, verose=False):
    # h_real, w_real = 312, 312     # original network input
    num_layers_ph = 2               # layer number for projection_head
    feat_dims_ph = [128, 64]
    save_path = 'models/trashnet'

    # base model
    params = {
        'input_shape': (80, 80, 3),
        'depth': 10,
        'width': 2,
        'drop_prob': 0.2,
        'out_units': 128
    }
    base_model = build_wide_resnet(params, has_top=False)
    dummy_in = np.random.rand(1, opt.height, opt.width, 3)
    dummy_out = base_model.predict(dummy_in)
    num_of_unfrozen_layers = 0          # 0: train all; >0: last layers

    print("base model extracted")
    base_model.summary()
    input_shape = (opt.height, opt.width, 3)
    feat_dim = np.prod(dummy_out.shape)
    print("output shape:", dummy_out.shape)

    # create SimCLR model
    SimCLR0 = SimCLR(
        base_model=base_model,
        input_shape=input_shape,
        batch_size=opt.batch_size,
        feat_dim=feat_dim,
        feat_dims_ph=feat_dims_ph,
        num_of_unfrozen_layers=num_of_unfrozen_layers,
        save_path=save_path
    )
    if verose:
        SimCLR0.SimCLR_model.summary()
    print("-> new SimCLR model created")
    test_w = 'models/test_weights.h5'
    SimCLR0.SimCLR_model.save_weights(test_w)
    SimCLR0.SimCLR_model.load_weights(test_w)
    os.remove(test_w)
    return SimCLR0


def load_dataset(opt):
    # read data
    df = pd.read_pickle(opt.split_filepath)
    df.head()
    class_labels = ["vehicle"]
    num_classes = len(df['class_one_hot'][0])
    print("# of training instances:", len(df.index), "\n")
    for label in class_labels:
        print(f"# of '{label}' training instances: {(df.class_label == label).sum()}")

    # prepare train and val data
    df_train, df_val_test = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
    df_val, df_test = train_test_split(df_val_test, test_size=0.50, random_state=42, shuffle=True)

    print("# of training instances:", len(df_train.index), "\n")
    for label in class_labels:
        print(f"# of '{label}' training instances: {(df_train.class_label == label).sum()}")
    print("# of validation instances:", len(df_val.index), "\n")
    for label in class_labels:
        print(f"# of '{label}' training instances: {(df_val.class_label == label).sum()}")
    print()
    print("# of test instances:", len(df_test.index), "\n")
    for label in class_labels:
        print(f"# of '{label}' training instances: {(df_test.class_label == label).sum()}")

    # data augmentation
    dfs = {
        "train": df_train,
        "val": df_val,
        "test": df_test
    }
    params_generator = {'batch_size': opt.batch_size,
                        'shuffle': True,
                        'width': opt.width,
                        'height': opt.height,
                        'VGG': False
                        }
    data_train = DataGenerator(df_train.reset_index(drop=True), **params_generator)
    data_val = DataGenerator(df_val.reset_index(drop=True), subset="val",
                             **params_generator)  # val keeps the unity values on the same random places ~42
    data_test = DataGenerator(df_test.reset_index(drop=True), subset="test",
                              **params_generator)  # test keeps the unity values on the diagonal
    return data_train, data_val, data_test


def start_training(opt):
    SimCLR0 = prepare_models(opt, verose=False)
    if opt.weights_path is not None:
        SimCLR0.SimCLR_model.load_weights(opt.weights_path)
        print("trained SimCLR model reloaded")
    else:
        print('train from scratch')
    # ph = extract_ph(SimCLR0.SimCLR_model)

    data_train, data_val, data_test = load_dataset(opt)
    print("data augmentation complete")
    pred_before = SimCLR0.predict(data_test)
    # test_nn(SimCLR0, data_test, opt.batch_size, y_test_before=pred_before)
    print("start training ...")
    SimCLR0.train(data_train, data_val, epochs=opt.num_epochs)
    # print("training complete")
    test_nn(SimCLR0, data_test, opt.batch_size, y_test_before=pred_before)
    # calc_feature_one_img(SimCLR1, image)


def test_nn(SimCLR0, data_test, batch_size, y_test_before=None):
    if y_test_before is not None:
        print('\n', f"accuracy - test minibatch: "
              f"{np.round(np.sum(data_test[0][1] * y_test_before[:batch_size]) / (2 * batch_size), 2)}")
        print("before training: ")
        for i in range(min(batch_size, 15)):
            print(np.round(y_test_before[i][i], 2), end=" | ")

    y_test_after = SimCLR0.predict(data_test)
    print('\n', f"accuracy - test minibatch: "
          f"{np.round(np.sum(data_test[0][1] * y_test_after[:batch_size]) / (2 * batch_size), 2)}")
    print("after training: ")
    for i in range(min(batch_size, 15)):
        print(np.round(y_test_after[i][i], 2), end=" | ")


def get_proj_model(SimCLR):
    out = SimCLR.SimCLR_model.get_layer(index=-3).output
    model = tf.keras.Model(SimCLR.SimCLR_model.inputs, out)
    model.summary()
    return model


def post_pool(x, pool_mode='mean'):
    if len(x.shape) == 2 and x.shape[0] == 1:
        return x

    if pool_mode == 'max':
        x = tf.reduce_max(x, (0, 1, 2), keepdims=True)
    elif pool_mode == 'mean':
        x = tf.reduce_mean(x, (0, 1, 2), keepdims=True)
    return tf.squeeze(x, (1, 2))


if __name__ == '__main__':
    app.run(main)



# def calc_feature_one_img(base_model, ph, img):
#     raw_feat = base_model.predict(img)
#     print("feature before projection:", raw_feat.shape)
#     feat = ph.predict(raw_feat)
#     print("feature after projection:", feat.shape)
#     return feat

# def SimCLR_no_CosSim(SimCLR):
#     feat_vec = SimCLR.get_layer(index=-2).output
#     print("layer name:", SimCLR.get_layer(index=-2).name)
#     print("feature vec shape:", feat_vec.shape)
#     feat_extractor = tf.keras.Model(SimCLR.inputs, feat_vec)
#     feat_extractor.summary()
#     return feat_extractor

# def extract_ph(SimCLR):
#     """ get GMA + projection head"""
#     input_l = tf.keras.Input(shape=(None, None, 1024))    # yolo intermediate output 13*13*1024
#     gma = tf.keras.layers.GlobalMaxPool2D()(input_l)    # flatten
#
#     proj_0 = SimCLR.get_layer(index=-3)(gma)
#     proj_1 = SimCLR.get_layer(index=-2)(proj_0)
#     ph = tf.keras.Model(input_l, proj_1)
#     save_path = './models/feat_ph.h5'
#     optimizer = Adam(1e-4, amsgrad=True)
#     loss = "categorical_crossentropy"
#     ph.compile(optimizer=optimizer, loss=loss)
#     ph.save(save_path, overwrite=True)
#     ph1 = tf.keras.models.load_model('./models/feat_ph.h5')
#     ph1.summary()
#     return ph1