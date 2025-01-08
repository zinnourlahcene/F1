import pandas as pd

from trainer.cnn.alexnet import alexNet
from trainer.cnn.googlenet import googLeNet
import cv2

EPOCHS = 10
MODEL_NAME = 'F1_model_bd_Vback5_speed.pkl'
CSV_SPEED_DATA_PATH = r'C:\Users\Dell\Desktop\F1\csv2\data_bd_speed.csv'

train_data = pd.read_csv(CSV_SPEED_DATA_PATH, header=None)
(images, speed) = (list(map(lambda featurs: cv2.imread(featurs, cv2.IMREAD_COLOR), train_data[0])),
                  list(map(lambda s1: [s1[0], s1[1],s1[2]], zip(train_data[3], train_data[4], train_data[5]))))


model = alexNet(5)
# model = resNext()
# model = googLeNet(3)

arr = (images, speed)
train = (images[0: int(len(arr[0]) * 0.8)], speed[0: int(len(arr[0]) * 0.8)])
test = (images[int(len(arr[0]) * 0.8): len(arr[0])], speed[int(len(arr[0]) * 0.8): len(arr[0])])

X = train[0]
Y = train[1]
print()

test_x = test[0]
test_y = test[1]

model.fit(X, Y, n_epoch=150, validation_set=(test_x, test_y),
   snapshot_epoch=False, snapshot_step=500,
   show_metric=True, batch_size=256, shuffle=True,
   run_id='alexnet')

# model.fit(X, Y, n_epoch=20, validation_set=(test_x, test_y), shuffle=True,
#           show_metric=True, batch_size=64, snapshot_step=200,
#           snapshot_epoch=False, run_id='googlenet')

# =============================================================================
#     model.fit(X, Y, n_epoch=10, validation_set=(test_x, test_y),
#           snapshot_epoch=False, snapshot_step=500,
#           show_metric=True, batch_size=128, shuffle=True,
#           run_id='resnext')
#
# =============================================================================
model.save(MODEL_NAME)
