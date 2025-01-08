# Author: Zinnour Lahcene
# ============================== loading libraries ===========================================
import numpy as np
import cv2
from PIL import ImageGrab as ig
import time
import os
import csv
from pynput import keyboard
from keys.getkeys import key_check, keys_to_output_movement
import joblib
from sklearn.metrics import accuracy_score

# =============================================================================================
#					                   Part I
# =============================================================================================

# ============================== data preprocessing ===========================================
paused = False

#local storage path
PATH_csv = r'C:\Users\Dell\Desktop\F1\csv2\data.csv'
PATH_knn = r'C:\Users\Dell\Desktop\F1\F1_CV\pkl\knn_speed_model.pkl'
PATH_digits = r'C:\Users\Dell\Desktop\F1\images\digits'
PATH_F1 = r'C:\Users\Dell\Desktop\F1\images\F1'
PATH_F1_local = r'C:\Users\Dell\Desktop\F1\images\F1'
PATH_speed = r'C:\Users\Dell\Desktop\F1\images\speed'
sep = '\\'
gs_sep = '/'
sep = gs_sep

#Google storage path
GS_F1 = "https://storage.googleapis.com/ultra-depot-244223/PIF6004/images/F1"
GS_speed = "https://storage.googleapis.com/ultra-depot-244223/PIF6004/images/speed"

PATH_F1_csv = GS_F1
PATH_speed_csv = GS_speed

nb = len(next(os.walk(PATH_F1_local))[2])
print(nb)

for i in list(range(3))[::-1]:
    print('.')
    time.sleep(1)

while True:
    if not paused:
        keys = key_check()
        if not keys:
            continue
        f = open(PATH_csv, 'a+', newline='')
        writer = csv.writer(f)

        csv_file_path = ''
        last_time = time.time()
        # ---------------------------------------------------------------------------------------------------------
        screen = ig.grab(bbox=(0, 30, 683, 728))
        # screen = ig.grab(bbox=(0, 0, 1366, 768))
        screen_np = np.array(screen)  # this is the array obtained from conversion
        screen_copy = screen_np
        screen_RGB = cv2.cvtColor(screen_copy, cv2.COLOR_BGR2RGB)
        hls = cv2.cvtColor(screen_RGB, cv2.COLOR_RGB2HLS)  # image convertion for better color detection (white color)
        # white color selection cv2.inRange une filtre pour detecter la couleur blanc,  cv2.bitwise_and retourner 255
        # si la couleur blanc est detecter
        white_images = cv2.bitwise_and(screen_copy, screen_copy,
                                       mask=cv2.inRange(hls, np.uint8([0, 200, 0]), np.uint8([255, 255, 255])))
        gray_images = cv2.cvtColor(np.array(white_images), cv2.COLOR_BGR2GRAY)
        blurred_images = cv2.GaussianBlur(gray_images, (15, 15), 0)
        edge_images = cv2.Canny(np.array(blurred_images), 5, 25)
        edge_images_copy = np.copy(edge_images)
        # ---------------------------------------------------------------------------------------------------------
        screen_speed = screen
        rows_speed, cols_speed = np.array(screen_speed).shape[:2]
        screen_speed = screen_speed.crop((cols_speed * 0.7795, rows_speed * 0.884, cols_speed * 0.8352857, rows_speed * 0.93))
        # speed_resize = cv2.resize(np.array(screen_speed), (60, 30))
        rows1, cols1 = np.array(screen_speed).shape[:2]
        digit1 = cv2.resize(np.array(screen_speed.crop((0, 0, cols1 * 0.333, rows1))), (20, 30))
        digit2 = cv2.resize(np.array(screen_speed.crop((cols1 * 0.333, 0, cols1 * 0.666, rows1))), (20, 30))
        digit3 = cv2.resize(np.array(screen_speed.crop((cols1 * 0.666, 0, cols1, rows1))), (20, 30))

        digits = [digit1, digit2, digit3]
        k = 0
        while k < len(digits):
            speed_screen_RGB = cv2.cvtColor(digits[k], cv2.COLOR_BGR2RGB)
            hls_speed = cv2.cvtColor(speed_screen_RGB, cv2.COLOR_RGB2HLS)
            white_images_speed = cv2.bitwise_and(digits[k], digits[k],
                                                 mask=cv2.inRange(hls_speed, np.uint8([0, 200, 0]),
                                                                  np.uint8([255, 255, 255])))
            gray_images_speed = cv2.cvtColor(np.array(white_images_speed), cv2.COLOR_BGR2GRAY)
            blurred_images_speed = cv2.GaussianBlur(gray_images_speed, (15, 15), 0)
            digits[k] = blurred_images_speed
            k += 1

        cv2.imwrite(os.path.join(PATH_digits, 'digit1.jpg'), digits[0])
        cv2.imwrite(os.path.join(PATH_digits, 'digit2.jpg'), digits[1])
        cv2.imwrite(os.path.join(PATH_digits, 'digit3.jpg'), digits[2])

        digit1 = np.append(cv2.imread(os.path.join(PATH_digits, 'digit1.jpg')), 0)
        digit1 = digit1.reshape(-1, 1)
        digit2 = np.append(cv2.imread(os.path.join(PATH_digits, 'digit2.jpg')), 0)
        digit2 = digit2.reshape(-1, 1)
        digit3 = np.append(cv2.imread(os.path.join(PATH_digits, 'digit3.jpg')), 0)
        digit3 = digit3.reshape(-1, 1)
        digits = [digit1, digit2, digit3]

        predicted_speed = ''
        knn = joblib.load(PATH_knn)
        for digit in digits:
            da = digit.reshape(1, -1)
            pred = knn.predict(da)
            # print(accuracy_score(digit, pred))
            predicted_speed += str(pred[0])
        d1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        d3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        d1[int(list(predicted_speed)[0])] = 1
        d2[int(list(predicted_speed)[1])] = 1
        d3[int(list(predicted_speed)[2])] = 1
        predicted_speed = int(predicted_speed)

        # ---------------------------------------------------------------------------------------------------------
        rows, cols = np.array(edge_images).shape[:2]
        A = [cols * 0, rows * 0.85]
        B = [cols * 0, rows * 0.6]
        C = [cols * 0.35, rows * 0.5]
        D = [cols * 0.65, rows * 0.5]
        E = [cols * 1, rows * 0.6]
        F = [cols * 1, rows * 0.8]
        G = [cols * 0.6, rows * 0.8]
        H = [cols * 0.6, rows * 0.55]
        I = [cols * 0.4, rows * 0.55]
        J = [cols * 0.4, rows * 0.85]
        vertices = np.array([[A, B, C, D, E, F, G, H, I, J]], dtype=np.int32)
        mask = np.zeros(np.array(edge_images).shape, dtype=np.uint8)
        if len(mask.shape) == 2:
            cv2.fillPoly(mask, vertices, 255)
        else:
            cv2.polylines(mask, vertices, (255,) * mask.shape[2])
        roi_images = cv2.bitwise_and(np.array(edge_images), mask)
        # ---------------------------------------------------------------------------------------------------------
        list_of_lines = cv2.HoughLinesP(roi_images, 1, np.pi / 180, 10, minLineLength=10, maxLineGap=250)
        # ---------------------------------------------------------------------------------------------------------
        rgb_copy = np.copy(screen_RGB)
        try:
            for line in list_of_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(rgb_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
        except Exception as e:
            print('There is no line to be detected!')
        # ---------------------------------------------------------------------------------------------------------

        output_movement = keys_to_output_movement(keys)
        resized_img = cv2.resize(np.array(rgb_copy), (227, 227))
        img_name = 'img{:>01}.jpg'.format(nb)
        cv2.imwrite(os.path.join(PATH_F1, img_name), resized_img)

        speed_img_name = img_name
        cv2.imwrite(os.path.join(PATH_speed, speed_img_name), np.array(screen_speed))

        writer.writerow((PATH_F1_csv + sep + img_name, PATH_speed_csv + sep + img_name, predicted_speed,
            d1, d2, d3,
            output_movement))

        if output_movement[3] == 1:
            nb += 1
            output_movement = [0, 0, 0, 0, 1]
            resized_img = cv2.flip(resized_img, 1)
            img_name = 'img{:>01}.jpg'.format(nb)
            cv2.imwrite(os.path.join(PATH_F1, img_name), resized_img)
            img_name_speed1 = 'img{:>01}.jpg'.format(nb-1)
            writer.writerow((
                PATH_F1_csv + sep + img_name, PATH_speed_csv + sep + img_name_speed1, predicted_speed,
                d1, d2, d3,
                output_movement))
            nb += 1
            img_name = 'img{:>01}.jpg'.format(nb)
            cv2.imwrite(os.path.join(PATH_F1, img_name), resized_img)
            img_name_speed2 = 'img{:>01}.jpg'.format(nb-2)
            writer.writerow((
                PATH_F1_csv + sep + img_name, PATH_speed_csv + sep + img_name_speed2, predicted_speed,
                d1, d2, d3,
                output_movement))
        elif output_movement[4] == 1:
            nb += 1
            output_movement = [0, 0, 0, 1, 0]
            resized_img = cv2.flip(resized_img, 1)
            img_name = 'img{:>01}.jpg'.format(nb)
            cv2.imwrite(os.path.join(PATH_F1, img_name), resized_img)
            img_name_speed1 = 'img{:>01}.jpg'.format(nb-1)
            writer.writerow((
                PATH_F1_csv + sep + img_name, PATH_speed_csv + sep + img_name_speed1, predicted_speed,
                d1, d2, d3,
                output_movement))
            nb += 1
            img_name = 'img{:>01}.jpg'.format(nb)
            cv2.imwrite(os.path.join(PATH_F1, img_name), resized_img)
            img_name_speed2 = 'img{:>01}.jpg'.format(nb-2)
            writer.writerow((
                PATH_F1_csv + sep + img_name, PATH_speed_csv + sep + img_name_speed2, predicted_speed,
                d1, d2, d3,
                output_movement))
        elif output_movement[2] == 1:
            nb += 1
            output_movement = [0, 0, 1, 0, 0]
            img_name = 'img{:>01}.jpg'.format(nb)
            cv2.imwrite(os.path.join(PATH_F1, img_name), resized_img)
            img_name_speed1 = 'img{:>01}.jpg'.format(nb-1)
            writer.writerow((
                PATH_F1_csv + sep + img_name, PATH_speed_csv + sep + img_name_speed1, predicted_speed,
                d1, d2, d3,
                output_movement))
            nb += 1
            img_name = 'img{:>01}.jpg'.format(nb)
            cv2.imwrite(os.path.join(PATH_F1, img_name), resized_img)
            img_name_speed2 = 'img{:>01}.jpg'.format(nb-2)
            writer.writerow((
                PATH_F1_csv + sep + img_name, PATH_speed_csv + sep + img_name_speed2, predicted_speed,
                d1, d2, d3,
                output_movement))
        # print('Loop took {} seconds',format(time.time()-last_time))
        # last_time = time.time()
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
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

