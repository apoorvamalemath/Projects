from sklearn.cluster import KMeans
from os.path import exists, isdir, basename, join, splitext
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
from pickle import dump, HIGHEST_PROTOCOL
import argparse
import numpy as np
from numpy import *
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from scipy import spatial
from iteration_utilities import deepflatten
#from scipy.stats import wasserstein_distance
import numpy, scipy.io
import re
import sys
import warnings
from sklearn.metrics.pairwise import euclidean_distances
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import pickle

EXTENSIONS = [".txt"]
PRE_ALLOCATION_BUFFER = 1000  


def kmeansclustering(data,n_clusters):
	kmeans = KMeans(n_clusters, random_state=0).fit(data)
	res=kmeans.labels_
	kmeans_center=[]
	kmeans_center=kmeans.cluster_centers_
	i=0
	descriptor_dim=100
	l2=[]
	signatures=[]
	leng=0
	for x in range(n_clusters):
		cluster_ele=[]
		for i in range(len(data)):
			if res[i]==x:
				cluster_ele.append(data[i])
		leng=leng+len(cluster_ele)	
		l1=[]
		for i in range(len(cluster_ele)):
			c=0	
			for j in range(descriptor_dim):	
				c=c+cluster_ele[i][j]		
			c=c/descriptor_dim
			l1.append(c)		

		no_des=len(cluster_ele) #number of descriptors in each cluster
		k=n_clusters    #number of clusters
    		#l1 has the cluster descriptors
		ci_s=0
		for i in range(len(l1)):
			ci_s=ci_s+l1[i]
	
		centroid_1=ci_s/no_des
		c=0
		for i in range(len(cluster_ele)):
			c=c+cluster_ele[i][0]

		centroid_2=c/no_des
	
		centroid_3=0
		
		for i in range(n_clusters):
			#print(kmeans_center[i])
			s=0
			for j in range(len(kmeans_center[i])):
				s=s+kmeans_center[i][j]
			s=s/descriptor_dim
			centroid_3=centroid_3+s
		


		w=float(no_des/k)
	
		l1=[]
		l1.append(centroid_1)
		l1.append(centroid_2)
		l1.append(centroid_3)
		l1.append(w)		
		signatures.append(l1)
		
	print(signatures)
	print("Len=", leng)
	with open('outfile', 'wb') as fp:
 	   	pickle.dump(signatures, fp)



def read_features(filename):
              
    f = open(filename, 'r')
    num = len(f.readlines())   # the number of features       
    featlength = 100            # the length of the descriptor
    f.close()

    f = open(filename, 'r')                 
    descriptors = np.zeros((num, featlength));        
    data = []
    lines = f.readlines()

    for line in lines:
        data.append([float(v) for v in line.split()])

    for i in range(num):
        for j in range(featlength):
            descriptors[i, j] = data[i][j]

        #normalize each input vector to unit length
        descriptors[i] = descriptors[i] / linalg.norm(descriptors[i])
        #print(descriptors[i])


    f.close()
    return descriptors	



if __name__ == '__main__':
    print("---------------------")
    print("Loading feature files")

    all_features_array = read_features(sys.argv[1])
    #print(all_features_array[0]) 
    nfeatures = all_features_array.shape[0]

    print("No of features %d" % int(nfeatures))

    N = int(nfeatures)
    #print(type(all_features_array))

    if N<=4000:
        clusters=1
    elif N<6000:
        clusters=3
    elif N<7000:
        clusters=4
    elif N<9000:
        clusters=12
    else:
        clusters=25
    kmeansclustering(all_features_array,clusters)



















    
