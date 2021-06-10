from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
import math


"""
k_len = []
for i in range(5, len(x)-6):
    x_ = (x[i] - x[i-5], y[i] - y[i-5])
    y_ = (x[i+5] - x[i], y[i+5] - y[i])
    d = 1 - pdist([x_, y_], 'cosine')
    sin = np.sqrt(1 - d**2)
    dis = np.sqrt((x[i-5] - x[i+5])**2 + (y[i-5] - y[i+5])**2)
    k = 2 * sin/dis
    print(k)
    k_len.append(k)
"""
def CircleFitting(x,y):
    u"""Circle Fitting with least squared
        input: point x-y positions
        output  cxe x center position
                cye y center position
                re  radius of circle
    """

    sumx  = sum(x)
    sumy  = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix,iy) in zip(x,y)])

    F = np.array([[sumx2,sumxy,sumx],
                  [sumxy,sumy2,sumy],
                  [sumx,sumy,len(x)]])

    G = np.array([[-sum([ix ** 3 + ix*iy **2 for (ix,iy) in zip(x,y)])],
                  [-sum([ix ** 2 *iy + iy **3 for (ix,iy) in zip(x,y)])],
                  [-sum([ix ** 2 + iy **2 for (ix,iy) in zip(x,y)])]])

    try:
        T=np.linalg.inv(F).dot(G)
    except:
        return (0,0,float("inf"))

    cxe=float(T[0]/-2)
    cye=float(T[1]/-2)
    #  print (cxe,cye,T)
    try:
        re=math.sqrt(cxe**2+cye**2-T[2])
    except:
        return (cxe,cye,float("inf"))
    return (cxe,cye,re)

def CalcCurvature(x,y,npo=1):
    u"""
    Calc curvature
    x,y: x-y position list
    npo: the number of points using Calculation curvature
    ex) npo=1: using 3 point
        npo=2: using 5 point
        npo=3: using 7 point
    """

    cv=[]
    ndata=len(x)
    for i in range(ndata):
        lind=i-npo
        hind=i+npo+1

        if lind<0:
            lind=0
        if hind>=ndata:
            hind=ndata
        #  print(lind,hind)

        xs=x[lind:hind]
        ys=y[lind:hind]
        #  print(xs,ys)
        (cxe,cye,re)=CircleFitting(xs,ys)
        #  print(re)

        if len(xs)>=3:
            # sign evalation
            cind=int((len(xs)-1)/2.0)
            sign = (xs[0] - xs[cind]) * (ys[-1] - ys[cind]) - (ys[0] - ys[cind]) * (xs[-1] - xs[cind])

            # check straight line
            a = np.array([xs[0]-xs[cind],ys[0]-ys[cind]])
            b = np.array([xs[-1]-xs[cind],ys[-1]-ys[cind]])
            try:
                theta=math.degrees(math.acos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b) + 0.0001)))
            except:
                theta = 0
            #  print(theta)

            if theta==180.0:
                cv.append(0.0)#straight line
            elif sign>0:
                cv.append(1.0/-(re+0.00001))
            else:
                cv.append(1.0/(re+ 0.00001))
        else:
            cv.append(0.0)

    #  print(cv)
    return cv



