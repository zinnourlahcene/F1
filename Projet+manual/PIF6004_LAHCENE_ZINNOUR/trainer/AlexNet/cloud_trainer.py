import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import cv2
from tensorflow.python.lib.io import file_io
import pandas as pd
from trainer.cnn.googlenet import googLeNet
from ast import literal_eval
import numpy as np
import tflearn
from skimage import io
import pickle
try:
    import urllib.request as urllib # for python 3
except ImportError:
    import urllib as urllib # for python 2



class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        self.val_acc_thresh = val_acc_thresh

    def on_epoch_end(self, training_state):
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration


if __name__ == '__main__':
    TRAIN_PATHS = 'gs://ultra-depot-244223/PIF6004'  # train_data = r"C:\Users\Dell\Desktop\F1" "gs://ultra-depot-244223/PIF6004"
    OUTPUT_DIR = 'gs://ultra-depot-244223/PIF6004/output'  # output = r"C:\Users\Dell\Desktop\F1\out" "gs://ultra-depot-244223/PIF6004/output"
    # python -m r"C:\Users\Dell\Desktop\F1\F1_CV\trainer\cloud_trainer.py" --train_data_paths r"C:\Users\Dell\Desktop\F1" --output_dir r"C:\Users\Dell\Desktop\F1\out"

    # gcloud ai-platform local train --package-path=\trainer\  --module-name=trainer.py --job-dir \out


    train_data = pd.read_csv(TRAIN_PATHS + '/csv/data.csv', header=None)

    (images, keys) = (list(map(
        lambda featurs: io.imread(featurs),train_data[0])),
        list(map(lambda labels: literal_eval(labels), train_data[6])))

    model = googLeNet(5)
    MODEL_NAME = 'model_keys'

    arr = (images, keys)
    train = (images[0: int(len(arr[0]) * 0.8)], keys[0: int(len(arr[0]) * 0.8)])
    test = (images[int(len(arr[0]) * 0.8): len(arr[0])], keys[int(len(arr[0]) * 0.8): len(arr[0])])


    X = train[0]
    Y = train[1]
    print(len(X))
    print(len(Y))
    test_x = test[0]
    test_y = test[1]
    # tensorboard = callbacks.TensorBoard(log_dir=TRAIN_PATHS + '/logs', histogram_freq=0, write_graph=True, write_images=False)
    # early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.32)

    model.fit(X, Y, n_epoch=300, validation_set=(test_x, test_y), shuffle=True,
              show_metric=True, batch_size=128, snapshot_step=200,
              snapshot_epoch=False, run_id='googlenet'
              )  # , callbacks=[tensorboard] , callbacks=early_stopping_cb

    model.save(MODEL_NAME)

    # Save model.h5 on to google storage
    with file_io.FileIO(MODEL_NAME+'.index', mode='r') as input_f_index:
        with file_io.FileIO(TRAIN_PATHS+'/' + MODEL_NAME+'.index', mode='w+') as output_f_index:
            output_f_index.write(input_f_index.read())

    with file_io.FileIO(MODEL_NAME+'.data-00000-of-00001', mode='r') as input_f_data:
        with file_io.FileIO(TRAIN_PATHS+'/' + MODEL_NAME+'.data-00000-of-00001', mode='w+') as output_f_data:
            output_f_data.write(input_f_data.read())

    with file_io.FileIO(MODEL_NAME+'.meta', mode='r') as input_f_meta:
        with file_io.FileIO(TRAIN_PATHS+'/' + MODEL_NAME+'.meta', mode='w+') as output_f_meta:
            output_f_meta.write(input_f_meta.read())


# gcloud ai-platform jobs submit training "job_pif_14" --package-path=trainer --module-name trainer.cloud_trainer --job-dir gs://ultra-depot-244223/PIF6
# 004/jobs --region us-east1 --config=trainer/cloudml-gpu.yaml


# gcloud ai-platform jobs submit training "job_pif_23" --package-path=trainer --module-name trainer.cloud_trainer --job-dir gs://ultra-depot-244223/PIF6
# 004/jobs --region us-east1