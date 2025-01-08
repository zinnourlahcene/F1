import pandas as pd
from trainer.cnn.googlenet import googLeNet
import cv2
from ast import literal_eval

EPOCHS = 1
MODEL_NAME = 'F1_model_bd_Vback4_mouvment.pkl'

train_data = pd.read_csv(r'C:\Users\Dell\Desktop\F1\csv2\data.csv', header= None)

(images, keys) = (list(map(lambda featurs: cv2.imread(featurs, cv2.IMREAD_COLOR), train_data[0])),
                  list(map(lambda labels: literal_eval(labels), train_data[3])))

#model = alexNet()
#model = resNext()
model = googLeNet(5)

arr = (images, keys)
train = (images[0 : int(len(arr[0]) * 0.8)], keys[0 : int(len(arr[0]) * 0.8)])
test = (images[int(len(arr[0]) * 0.8) : len(arr[0])], keys[int(len(arr[0]) * 0.8) : len(arr[0])])

X = train[0]
Y = train[1]
print()

test_x = test[0]
test_y = test[1]

model.fit(X, Y, n_epoch=5, validation_set=(test_x, test_y),
   snapshot_epoch=False, snapshot_step=500,
   show_metric=True, batch_size=256, shuffle=True,
   run_id='alexnet')


# model.fit(X, Y, n_epoch=30, validation_set=(test_x, test_y), shuffle=True,
#     show_metric=True, batch_size=64, snapshot_step=200,
#     snapshot_epoch=False, run_id='googlenet')

# =============================================================================
#     model.fit(X, Y, n_epoch=10, validation_set=(test_x, test_y),
#           snapshot_epoch=False, snapshot_step=500,
#           show_metric=True, batch_size=128, shuffle=True,
#           run_id='resnext')
#
# =============================================================================
model.save(MODEL_NAME)
