import numpy as np
from grabscreen import grab_screen
import cv2
import time
from keys.directkeys import PressKey, ReleaseKey, SPACE, UP, DOWN, RIGHT, LEFT
from keys.getkeys import key_check
from trainer.cnn.alexnet import alexNet
from trainer.cnn.googlenet import googLeNet

MODEL_NAME = 'model'


def straight():
    PressKey(SPACE)
    ReleaseKey(LEFT)
    ReleaseKey(RIGHT)
    ReleaseKey(UP)
    ReleaseKey(DOWN)


def back():
    PressKey(DOWN)
    ReleaseKey(SPACE)
    ReleaseKey(LEFT)
    ReleaseKey(RIGHT)
    ReleaseKey(UP)


def left():
    PressKey(LEFT)
    ReleaseKey(SPACE)
    ReleaseKey(RIGHT)
    ReleaseKey(UP)
    ReleaseKey(DOWN)


def right():
    PressKey(RIGHT)
    ReleaseKey(LEFT)
    ReleaseKey(SPACE)
    ReleaseKey(UP)
    ReleaseKey(DOWN)


# model = alexNet(5)
model = googLeNet(5)
model.load(MODEL_NAME)


def main():
    for i in list(range(3))[::-1]:
        print('.')
        time.sleep(1)

    paused = False
    while (True):

        if not paused:
            screen = grab_screen(region=(0, 30, 683, 728))
            screen_np = np.array(screen)
            screen_RGB = cv2.cvtColor(screen_np, cv2.COLOR_BGR2RGB)
            hls = cv2.cvtColor(screen_RGB, cv2.COLOR_RGB2HLS)
            white_images = cv2.bitwise_and(screen_np, screen_np,
                                           mask=cv2.inRange(hls, np.uint8([0, 200, 0]), np.uint8([255, 255, 255])))
            gray_images = cv2.cvtColor(np.array(white_images), cv2.COLOR_BGR2GRAY)
            blurred_images = cv2.GaussianBlur(gray_images, (15, 15), 0)
            edge_images = cv2.Canny(np.array(blurred_images), 5, 25)
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
            list_of_lines = cv2.HoughLinesP(roi_images, 1, np.pi / 180, 10, minLineLength=10, maxLineGap=250)
            rgb_copy = np.copy(screen_RGB)
            try:
                for line in list_of_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(rgb_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
            except Exception as e:
                print('There is no line to be detected!')
            resized_img = cv2.resize(np.array(rgb_copy), (227, 227))

            # =====================================================================================================

            prediction = model.predict([resized_img.reshape(227, 227, 3)])[0]
            # print(prediction)
            moves = list(np.around(prediction))
            # print(moves)

            if moves[0] == 1:
                print('| F |')
                straight()
            if moves[2] == 1:
                print('| B |')
                back()
            elif moves[3] == 1:
                print('          ==>>')
                right()
            elif moves[4] == 1:
                print('          <<==')
                left()
        #            else:
        #                print('| ? | ... ')
        #                print(moves)
        # straight()

        keys = key_check()

        # p pauses game and can get annoying.
        if 'P' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(SPACE)
                ReleaseKey(UP)
                ReleaseKey(DOWN)
                ReleaseKey(RIGHT)
                ReleaseKey(DOWN)
                time.sleep(1)


main()
