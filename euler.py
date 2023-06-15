import numpy as np

def euler(init, vector_field, h, N):
    y = np.zeros([N+1])
    print(vector_field(1., 0.))
    y[0] = init
    for k in range(1, N+1):
        y[k] = y[k-1] + h*vector_field(y[k-1], (k-1)*h)
        print(y[k])
    return y
