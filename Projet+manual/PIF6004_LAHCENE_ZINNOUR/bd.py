import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle

CSV_DATA_PATH = r'C:\Users\Dell\Desktop\F1\csv2\data_local.csv'
CSV_BALANCED_DATA_PATH = r'C:\Users\Dell\Desktop\F1\csv2\data_local_bd.csv'


train_data = pd.read_csv(CSV_DATA_PATH)
    
    
lefts = []
rights = []
forwards = []
backs = []

train_data = shuffle(train_data)

s = '[1, 0, 0, 0, 0]'
r = '[0, 0, 0, 1, 0]'
l = '[0, 0, 0, 0, 1]'
b = '[0, 0, 1, 0, 0]'

for index, data in train_data.iterrows():
    img = data[0]
    speed_img = data[1]
    speed = data[2]
    digit1 = data[3]
    digit2 = data[4]
    digit3 = data[5]
    choice = data[6]
  
    if choice == s:
        forwards.append([img,speed_img,speed,digit1,digit2,digit3,choice])
    elif choice == r:
        rights.append([img,speed_img,speed,digit1,digit2,digit3,choice])
    elif choice == l:
        lefts.append([img,speed_img,speed,digit1,digit2,digit3,choice])
    elif choice == b:
        backs.append([img,speed_img,speed,digit1,digit2,digit3,choice])
    else:
        print('no matches')

print(len(forwards))
print(len(rights))
print(len(lefts))
print(len(backs))
        
forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]


final_data = forwards + lefts + rights
final_data = shuffle(final_data)
df = pd.DataFrame(final_data)

df.to_csv(CSV_BALANCED_DATA_PATH, index = None, header=False)