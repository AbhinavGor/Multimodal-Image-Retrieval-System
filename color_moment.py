import torch, cv2, math, os, pandas, time, logging, numpy
from PIL import Image, ImageStat
from torchvision import transforms

from database_connection import connect_to_mongo

def extract_color_moment(img):
    # Separating three channels for easier calculation
    R, G, B = img[0], img[1], img[2]

    R_bin = [[[] for i in range(10)] for i in range(10)]
    G_bin = [[[] for i in range(10)] for i in range(10)]
    B_bin = [[[] for i in range(10)] for i in range(10)]
    
    # Binning all three channels separately into a 10x10 grid
    for i in range(len(R)):
        for j in range(len(R[i])):
            R_bin[int(i/10)][int(j/30)].append(R[i][j])

    for i in range(len(G)):
        for j in range(len(G[i])):
            G_bin[int(i/10)][int(j/30)].append(G[i][j])

    for i in range(len(B)):
        for j in range(len(B[i])):
            B_bin[int(i/10)][int(j/30)].append(B[i][j])

    # Defininig an emppty array for storing computed color moments
    color_moments = [[0 for i in range(10)] for i in range(10)]

    # Color moment computation
    for i in range(10):
        for j in range(10):
            sum_R = sum(R_bin[i][j])
            sum_G = sum(G_bin[i][j])
            sum_B = sum(B_bin[i][j])
            avg_R = sum_R/300
            avg_G = sum_G/300
            avg_B = sum_B/300
            sdev_R = (sum([(i - avg_R)**2 for i in R_bin[i][j]])/300)**(0.5)
            sdev_G = (sum([(i - avg_G)**2 for i in G_bin[i][j]])/300)**(0.5)
            sdev_B = (sum([(i - avg_B)**2 for i in B_bin[i][j]])/300)**(0.5)
            skew_R = sum([(i - avg_R)**3 for i in R_bin[i][j]])/300
            skew_R = numpy.sign(skew_R)*((numpy.abs(skew_R))**(1./3.))
            skew_G = sum([(i - avg_R)**3 for i in R_bin[i][j]])/300
            skew_G = numpy.sign(skew_G)*((numpy.abs(skew_G))**(1./3.))
            skew_B = sum([(i - avg_R)**3 for i in R_bin[i][j]])/300
            skew_B = numpy.sign(skew_B)*((numpy.abs(skew_B))**(1./3.))

            color_moments[i][j] = [[avg_R, sdev_R, skew_R], [avg_G, sdev_G, skew_G], [avg_B, sdev_B, skew_B]]
    return color_moments
    