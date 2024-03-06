import numpy as np

def string_pluck(U, x_num, pos, height):
    '''
    Takes U and populates it with the displacement at the 1st and 2nd time steps for a pluck at position pos with height height
    '''
    if not x_num % 2:
        for i in range(x_num + 1):
            if i == 0 or i == (x_num):
                U[0][i] = 0
            elif i< int(pos*x_num + 0.5):
                U[0][i] = U[0][i-1] + height /x_num / pos
            elif i > int(pos*x_num + 0.5):
                U[0][i] = U[0][i-1] - height/x_num / (1 - pos)
            else: U[0][i] = U[0][i-1]
    else:
        for i in range(x_num + 1):
                if i == 0 or i == (x_num):
                    U[0][i] = 0
                elif i<= int(pos*x_num + 0.5):
                    U[0][i] = U[0][i-1] + height /x_num / pos
                else:
                    U[0][i] = U[0][i-1] - height/x_num / (1 - pos)
    U[1] = U[0].copy()



def string_strike(U, x_num, pos, width, velocity, k):
    '''
    Takes U and populates it wit the displacement at the 1st and 2nd time steps for a simulated striking force
    '''
    V = np.zeros(x_num + 1)
    for i in range(x_num + 1):
        if int(x_num * (pos - width/2)) <= i <= int(x_num * (pos + width/2)):
            V[i] = velocity * np.sin(np.pi / int(x_num * width) * (i - int(x_num * (pos - width/2))))
        else:
            V[i] = 0
    U[1] = -V.copy() * k

