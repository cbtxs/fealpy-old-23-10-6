import numpy as np

def dcircle(p, cxy=[0, 0], r=1):
    x = p[:, 0]
    y = p[:, 1]
    return np.sqrt((x - cxy[0])**2 + (y - cxy[1])**2) - r

def drectangle(p, box):
    return -dmin(dmin(dmin(p[:, 1] - box[2], box[3]-p[:,1]), p[:,0] - box[0]), box[1] - p[:,0])

def dsine(p, cxy, r):
    x = p[:,0]
    y = p[:,1]
    return (y - cxy[1]) - r*np.sin(x-cxy[0])

def dparabolic(p,cxy,r):
    x = p[:,0]
    y = p[:,1]
    return (y - cxy[1])**2 - 2*r*x


def dpoly(p, poly):
    pass

def ddiff(d0, d1):
    return dmax(d0, -d1)

def dmin(d0, d1):
    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
    return dd.min(axis=1)

def dmax(d0, d1):
    dd = np.concatenate((d0.reshape((-1,1)), d1.reshape((-1,1))), axis=1)
    return dd.max(axis=1)