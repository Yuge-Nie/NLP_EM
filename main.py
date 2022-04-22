# -*- coding: utf-8 -*-
from random import random
import numpy as np
import math
import matplotlib.pyplot as plt

isdebug = True

def init_data(s1, s2, p, q, r, n):
    data = []
    for i in range(n):
        coin = random()
        if 0 <= coin < s1:
            side = np.random.binomial(1, p)
        elif s1 <= coin < s1 + s2:
            side = np.random.binomial(1, q)
        else:
            side = np.random.binomial(1, r)
        data.append(side)
    if isdebug:
        print("***********")
        print("观测数据：")
        print(data)
    return data

def EM(theta, y, iter_num, epsilon):
    s1 = theta[0]
    s2 = theta[1]
    p  = theta[2]
    q  = theta[3]
    r  = theta[4]
    n  = len(y)
    i = 0
    threshold = 99999
    while(i < iter_num and threshold >= epsilon):
        # E_Step
        a = np.random.rand(n)
        b = np.random.rand(n)
        for j in range(n):
            a[j] = (s1*pow(p,y[j])*pow(1-p,1-y[j]))/(s1*pow(p,y[j])*pow(1-p,1-y[j])+s2*pow(q,y[j])*pow(1-q,1-y[j])+(1-s1-s2)*pow(r,y[j])*pow(1-r,1-y[j]))
            b[j] = (s2*pow(q,y[j])*pow(1-q,1-y[j]))/(s1*pow(p,y[j])*pow(1-p,1-y[j])+s2*pow(q,y[j])*pow(1-q,1-y[j])+(1-s1-s2)*pow(r,y[j])*pow(1-r,1-y[j]))
        # M_Step
        s1_next = 1/n * sum(a)
        s2_next = 1/n * sum(b)
        p_next = sum([a[j]*y[j] for j in range(n)]) / sum(a)
        q_next = sum([b[j]*y[j] for j in range(n)]) / sum(b)
        r_next = sum([(1-a[j]-b[j])*y[j] for j in range(n)]) / sum([(1-a[j]-b[j]) for j in range(n)])
        # Threshold
        threshold = np.linalg.norm(np.array([s1-s1_next,s2-s2_next,p-p_next,q-q_next,r-r_next]), ord = 2)
        s1 = s1_next
        s2 = s2_next
        p  = p_next
        q  = q_next
        r  = r_next
        i += 1
        print(i, [s1, s2, p, q, r])
    return s1, s2, p, q, r


if __name__ == '__main__':
    y = init_data(0.1, 0.6, 0.8, 0.2, 0.3, 50)
    theta = [0.3, 0.3, 0.2, 0.5, 0.6]
    EM(theta, y, 100, 1e-21)
    # plt.hist(X[0, :], 50)
    # plt.show()

