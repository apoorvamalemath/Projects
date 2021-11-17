import numpy as np
from numpy import *
import pickle
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
#from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import manhattan_distances
from scipy import spatial
from iteration_utilities import deepflatten
import numpy, scipy.io
import sys
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':

    distance = sys.argv[1]

    with open("encoding.txt", "rb") as fp:   
        l2 = pickle.load(fp)

    #print("L2:", l2[708])

    dist_file = open("distance.txt", "w")

    if(distance == "Euclidean"):
        dist = []
        for i in range(1,len(l2)):
            q = []
            q.append(l2[i])
            #print(q)
            d = euclidean_distances(l2[1:], q)
            dist.append(d)

        for i in range(len(dist)):
            for j in range(len(dist)):
                dist_file.write("%f " % dist[i][j][0])
            dist_file.write("\n")

    elif(distance == "EMD"):
        dist = []
        for i in range(1,len(l2)):
            for j in range(1,len(l2)):        
                d = wasserstein_distance(l2[j],l2[i])
                dist.append(d)

        for i in range(len(dist)):
            dist_file.write("%f " % dist[i])
            if ((i+1) % (len(l2)-1) == 0):
                dist_file.write("\n")

    elif(distance == "ChiSquare"):
        dist = []
        for i in range(1,len(l2)):
            for j in range(1,len(l2)):        
                d = 0
                for k in range(1,len(l2[j])):
                    sum1 = l2[j][k] + l2[i][k]	
                    diff = l2[j][k] - l2[i][k]
                    if( (sum1 > 0) and (diff != 0)):
                        d = d + (diff*diff)/sum1;
                dist.append((1/2)*d)

        for i in range(len(dist)):
            dist_file.write("%f " % dist[i])
            if ((i+1) % (len(l2)-1) == 0):
                dist_file.write("\n")

    elif(distance == "Cosine"):
        dist = []
        for i in range(len(l2)):
            for j in range(len(l2)):        
                d = spatial.distance.cosine(l2[j],l2[i])
                dist.append(d)

        for i in range(len(dist)):
            dist_file.write("%f " % dist[i])
            if ((i+1) % (len(l2)-1) == 0):
                dist_file.write("\n")

    elif(distance == "L1"):
        dist = []
        for i in range(1,len(l2)):
            q = []
            q.append(l2[i])
            d = manhattan_distances(l2[1:], q)
            dist.append(d)

        for i in range(len(dist)):
            for j in range(len(dist)):
                dist_file.write("%f " % dist[i][j][0])
            dist_file.write("\n")

    
    print("---------------------")
