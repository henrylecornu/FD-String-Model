import numpy as np

def guitar_backboard(x_num, action):
    '''
    Returns a discrete model of the guitar backboard
    '''
    BB = np.zeros(x_num + 1)
    for i in range(x_num + 1):
        if i <= int(x_num * 0.25):
            BB[i] = -action * 10
        else:
            BB[i] = -action
    return BB

def sitar_bridge(x_num, L, height, len):
    '''
    Returns a discrete model of the sitar bridge
    '''
    BB = np.zeros(x_num + 1)
    for i in range(x_num + 1):
        j = (i / x_num) * L
        if j > len:
            BB[i] = -0.5
        else:
            BB[i] = -0.01 * j ** 2 / (L ** 2)
    return BB