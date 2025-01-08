import numpy as np
import cv2
from PIL import ImageGrab as ig
import time
import os
import csv
from keys.getkeys import key_check, keys_to_output_movement

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#local storage paths
CSV_DATA_PATH = r'C:\Users\Dell\Desktop\F1\csv2\data.csv'
DATA_PATH = r'C:\Users\Dell\Desktop\F1\img2'

nb=len(next(os.walk(DATA_PATH))[2])
print(nb)

paused = False

for i in list(range(3))[::-1]:
    print('.')
    time.sleep(1)
        
        
while True:
    if not paused:
        keys = key_check()
        if not keys:
            continue
        f= open(CSV_DATA_PATH,  'a+', newline='')
        writer = csv.writer(f)
    
        csv_file_path = ''
        last_time = time.time()
        #---------------------------------------------------------------------------------------------------------
        screen = ig.grab(bbox=(0,0,1366,768))
        screen_np = np.array(screen)
        
        rows, cols = np.array(screen).shape[:2]
        crop = screen.crop((cols * 0.7795, rows * 0.884, cols * 0.8352857, rows * 0.93))

        rows1, cols1 = np.array(crop).shape[:2]
        im = np.array(crop)
        digit1 = cv2.resize(np.array(crop.crop((0, 0, cols1*0.333, rows1))), (20, 30))
        digit2 = cv2.resize(np.array(crop.crop((cols1*0.333, 0, cols1*0.666, rows1))), (20, 30))
        digit3 = cv2.resize(np.array(crop.crop((cols1*0.666, 0, cols1, rows1))), (20, 30))
        digits = [digit1, digit2, digit3]
        k = 0
        while k < len(digits):
            screen_RGB = cv2.cvtColor(digits[k], cv2.COLOR_BGR2RGB)
            hls = cv2.cvtColor(screen_RGB, cv2.COLOR_RGB2HLS)
            white_images = cv2.bitwise_and(digits[k], digits[k], mask = cv2.inRange(hls, np.uint8([0,200,0]), np.uint8([255, 255, 255])))        
            gray_images = cv2.cvtColor(np.array(white_images), cv2.COLOR_BGR2GRAY)
            blurred_images =  cv2.GaussianBlur(gray_images, (15, 15), 0)
            edge_images = cv2.Canny(np.array(blurred_images), 5, 25)
            digits[k] = blurred_images
            k += 1


        #---------------------------------------------------------------------------------------------------------
        output_movement = keys_to_output_movement(keys)
        img_name = 'img{:>01}.jpg'.format(nb)
        
        cv2.imwrite(os.path.join(DATA_PATH, 'digit1_{:>01}.jpg'.format(nb)), digits[0])
        cv2.imwrite(os.path.join(DATA_PATH, 'digit2_{:>01}.jpg'.format(nb)), digits[1])
        cv2.imwrite(os.path.join(DATA_PATH, 'digit3_{:>01}.jpg'.format(nb)), digits[2])
        writer.writerow((DATA_PATH + '\\' + img_name, output_movement))
        
        
        nb += 1
        f.close()
    keys = key_check()
    if 'P' in keys:
        if paused:
            paused = False
            print('unpaused!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            time.sleep(1)
    elif 'Q' in keys:
        print('Quitting!')
        break
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
