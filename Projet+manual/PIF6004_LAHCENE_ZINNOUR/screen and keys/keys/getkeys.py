# Citation: Box Of Hats (https://github.com/Box-Of-Hats )
import win32con as wcon
import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    if wapi.GetAsyncKeyState(wcon.VK_UP):
        keys.append('up')
    if wapi.GetAsyncKeyState(wcon.VK_RCONTROL):
        keys.append('down')
    if wapi.GetAsyncKeyState(wcon.VK_RIGHT):
        keys.append('right')
    if wapi.GetAsyncKeyState(wcon.VK_LEFT):
        keys.append('left')
    if wapi.GetAsyncKeyState(wcon.VK_SPACE):
        keys.append('space')

    return keys


def keys_to_output_movement(keys):
    """
    Convert keys to a ...multi-hot... array
    ['space - 0', 'up - 1', 'down - 2', 'right - 3', 'left - 4']
    """
    output = [0, 0, 0, 0, 0]

    if 'space' in keys:
        output[0] = 1
    elif 'up' in keys:
        output[1] = 1
    elif 'down' in keys:
        output[2] = 1
    elif 'right' in keys:
        output[3] = 1
    elif 'left' in keys:
        output[4] = 1

    return output