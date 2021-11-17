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


def kmeansclustering(data):
	print(data)
	#kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
	#print(kmeans.labels_)
	#print(kmeans.cluster_centers_)
	#plt.imshow(kmeans)
    	#plt.show()
	


def adaptive_cluster(data, gap_par = 0.2, n0=None,debug=False,assign_outliers = 'nearest_cluster'):
    '''
    data:: a numeric numpy array
    gap_par: the lambda parameter used to test the gap
    n0: the initial neighbors for each data point. 
    debug: for debug
    assign_outliers: nearest_cluster, assign outliers to nearest cluster. new_cluster, assign outliers to a new cluster
    '''
    weight_matrix_history = []
    (n_points,n_features) = data.shape

    distance_matrix = scipy.spatial.distance.cdist(data, data, 'euclidean')
    #print('distance_matrix.shape',distance_matrix.shape)
    weight_matrix = np.zeros(shape=(n_points,n_points))
    weight_matrix_history.append((0,weight_matrix))
    
    ### sort the distance matrix
    sorted_distance_idx_matrix = np.argsort(distance_matrix,axis=1)
    sorted_distance_matrix = np.sort(distance_matrix,axis=1)


    ### number of neighbors
    if n0 is None:
        n0 = 2*n_features+2
    #print(n_features)
    
    ### h0 is the the radius such that the point has n0 neighbors 
    h0 = sorted_distance_matrix[:,n0]
    

    ### faster version
    h0_matrix = np.tile(h0, (n_points, 1))
    h0_matrix_T = h0_matrix.T
    h0_matrix_max = np.maximum(h0_matrix,h0_matrix_T)
    weight_matrix = (distance_matrix<=h0_matrix_max).astype('int')


    #################################################################
    ### find h sequence
    a = 1.4142135623730951
    b = 1.95
    max_distance = np.max(sorted_distance_matrix)

    ### h0 is a vector, each data point has n0 neighbors
    ### max(h0) makes sure that each data point has at least n0 neighbors
    h_array  = np.array([np.max(h0)])

    k = 0
    weight_matrix_history.append((h_array[k],weight_matrix.copy()))
    while h_array[k] <= max_distance:
        ### upper bound of n(Xi,h_k+1)
        ### given radius h_array[k], how many neighbors for each data point
        ### -1 removes its self from counting
        n_upper = a * np.array([np.sum(sorted_distance_matrix[i,:]<=h_array[k])-1 for i in np.arange(n_points)])
        n_upper = (np.floor(n_upper)).astype('int')
        ### when h is big, the n_upper may be > n_points
        n_upper = np.clip(n_upper, a_min=None,a_max=(n_points-1))
        #print(n_upper)
        ### n_upper can decide the h_upper
        h_upper_by_n_upper = np.min(np.array([sorted_distance_matrix[i,n_upper[i]] for i in np.arange(n_points)]))
        ### upper bound of h_k+1
        h_upper = b*h_array[k]
        ### must satisfy both conditions
        min_h_upper = np.minimum(h_upper_by_n_upper,h_upper)
        #print(k,min_h_upper)
        ### append to the h_array
        ### just make sure h is not > max_distance
        if min_h_upper <= max_distance:
            if  min_h_upper <= h_array[k]: break
            #print(k,'h',min_h_upper)
            h_array = np.append(h_array,min_h_upper)
        k = k + 1

    #################################################################    
    ### check if those h satisfy the conditions
    if debug:
        for k in range(1,len(h_array)):  
            if h_array[k] <= b*h_array[k-1]:
                continue
                print('k',k,h_array[k],h_array[k-1],b*h_array[k-1],',')
                print(h_array[k]/h_array[k-1])
            else:
                print('h error')
        for k in range(1,len(h_array)):
            for i in range(n_points):
                n1 = np.sum(sorted_distance_matrix[i,:]<=h_array[k-1])-1 
                n2 = np.sum(sorted_distance_matrix[i,:]<=h_array[k])-1 
                if n2<=a*n1 and n1>=n0 and n2>=n0:
                    continue
                    print('n',k,n1,n2,a*n1,',')
                    print(n2/n1)
                else:
                    print('n error')
        
    #################################################################
    beta_a = (n_features+1.0)/2.0
    #beta_b = 0.5
    beta_b = 0.4
    beta_function = scipy.special.beta(beta_a,beta_b)

    np.seterr(divide='ignore', invalid='ignore')  
    print('h_k',h_array[0])
    for k in range(1,len(h_array)):
        print('h_k',h_array[k])
        for i in range(n_points):
            weight_matrix[i,i] = 1
            for j in range(i,n_points):
                if distance_matrix[i,j] <= h_array[k] and h_array[k-1] >= h0[i] and h_array[k-1] >= h0[j]:
                    #### caclulate overlap
                    N_overlap = np.dot(weight_matrix[i,:],weight_matrix[j,:])
                    #### caclulate complement
                    if k>1:
                        ind1 = (distance_matrix[j,:] > h_array[k-1]) + 0.0
                        ind2 = (distance_matrix[i,:] > h_array[k-1]) + 0.0
                    else:
                        ind1 = (distance_matrix[j,:] > h0_matrix_max[i,j]) + 0.0
                        ind2 = (distance_matrix[i,:] > h0_matrix_max[i,j]) + 0.0
                    N_complement = np.dot(weight_matrix[i,:],ind1) + np.dot(weight_matrix[j,:],ind2)
                    #### caclulate union
                    N_union = N_overlap + N_complement
                    #### theta
                    theta = N_overlap / N_union
                    #### q
                    t = distance_matrix[i,j]/h_array[k-1]
                    beta_x = 1.0-(t**2)/4.0
                    incomplete_beta_function = scipy.special.betainc(beta_a,beta_b,beta_x)
                    q = incomplete_beta_function / (2*beta_function-incomplete_beta_function)
                    T1 = N_union
                    #### this may raise warnings about log(0) or log(nan)
                    #### this is fine, since I used the whole matrix here
                    #### some of the points are out of the h(k) radius
                    #### we will mask those points in the later step
                    T2 = theta*np.log(theta/q)+(1.0-theta)*np.log((1.0-theta)/(1.0-q))
                    #### when N_overlap is 0, theta is 0, this leands to T is nan
                    #### replace those nan with 0 in T
                    #T2 = np.where(theta==0.0,0.0,T2)
                    #T2 = np.where(theta==1.0,0.0,T2)
                    #T3 = ((theta<=q).astype('int')-(theta>q).astype('int'))
                    ### faster version
                    if theta<=q:
                        T = T1 * T2
                    else:
                        T = - (T1 * T2)
                    weight_matrix[i,j] = (T<=gap_par) + 0.0
                    #### be careful with those boundary points
                    #### theta=0 means no overlap at all
                    #### theta=1 means completely overlap
                    #### needs special treatment for them
                    if theta==0: weight_matrix[i,j] = 0 
                    if theta==1: weight_matrix[i,j] = 1
                    ####
                    weight_matrix[j,i] = weight_matrix[i,j]
        weight_matrix_history.append((h_array[k],weight_matrix.copy()))
        
    ### reset to default
    np.seterr(divide='warn', invalid='warn')  
    
    ### calculate S
    S = np.sum(weight_matrix)
    
    ### extract clusters from weight matrix
    labels = (np.zeros(shape=weight_matrix.shape[0]))
    labels.fill(np.nan)
    cluster_ind = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:continue
            if weight_matrix[i,j] == 1:
                if np.isnan(labels[i]) and np.isnan(labels[j]):
                    labels[i] = cluster_ind
                    labels[j] = cluster_ind
                    cluster_ind = cluster_ind + 1
                elif not np.isnan(labels[i]) and np.isnan(labels[j]):
                    labels[j] = labels[i]
                elif  np.isnan(labels[i]) and not np.isnan(labels[j]):
                    labels[i] = labels[j]
                elif  not np.isnan(labels[i]) and  not np.isnan(labels[j]):
                    continue
                else:
                    print(i,j,labels[i],labels[j])
                    print('cluster assignment error')
    ### some points may not belong to any cluster
    ### assign those points to the nearest cluster
    ### or they can be ignored (by default, those points will have np.nan as labels)
    ### thus those points can be considered as outliers
    if assign_outliers == 'nearest_cluster':
        if np.sum(np.isnan(labels))>0:
            nan_ind = np.argwhere(np.isnan(labels)).flatten()
            for i in nan_ind:
                dist = distance_matrix[i,:].copy()
                dist[i] = np.max(dist)
                nearest_ind = np.argmin(dist)
                labels[i] = labels[nearest_ind]
                #print(dist)
                #print(i,nearest_ind)
    elif assign_outliers == 'new_cluster':
        if np.sum(np.isnan(labels))>0:
            nan_ind = np.argwhere(np.isnan(labels)).flatten()
            outlier_label = np.nanmax(np.unique(labels)) + 1
            for i in nan_ind:
                labels[i] = outlier_label
    else:
        print('assign_outliers parameter is not correct')
    #print("cluster_ind",cluster_ind)       
    return({"S":S,"weight_matrix":weight_matrix,
            "cluster_label":labels,
            "weight_matrix_history":weight_matrix_history,
           })
               

def plot_weight_matrix(weight_matrix):
    plt.imshow(weight_matrix)
    plt.show()



def get_files(path):
    all_files = []
    all_files.extend([join(path, basename(fname))
                    for fname in glob(path + "/*")
                    if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files

def read_file(input_files):
    all_features_dict = {}
    for i, fname in enumerate(input_files):
        features_fname = fname
        descriptors = read_features(features_fname)
        all_features_dict[fname] = descriptors
    return all_features_dict

def read_features(filename):
              
    f = open(filename, 'r')
    num = len(f.readlines())   # the number of features       
    featlength = 31            # the length of the descriptor
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

def dict2numpy(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 31))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, 31))
    return array



if __name__ == '__main__':
    print("---------------------")
    print("Loading feature files")
    all_features_array=read_features("D00002.txt")
    nfeatures = all_features_array.shape[0]

    print("No of features %d" % int(nfeatures))

    
    N = int(nfeatures)
    print(type(all_features_array))

    result=adaptive_cluster(all_features_array)
    print('Number of clusters',np.unique(result['cluster_label']))
    
    labels=[]
    signatures=[]
    labels=np.unique(result['cluster_label'])
    res=result['cluster_label']
    #print(res)
    #print(labels)
    #print(all_features_array.shape)
    #print(type(result['cluster_label']))
    #print(result['cluster_label'].shape)
    i=0
    l2=[]
    for x in labels:
	cluster_ele=[]
	for i in range(len(all_features_array)):
		if res[i]==x:
			cluster_ele.append(all_features_array[i])
	l1=[]
	ci=[]
	for i in range(31):
		ci.append(0)
	#print(ci)
	for i in range(len(cluster_ele)):
		for j in range(31):
			ci[j]=ci[j]+cluster_ele[i][j]
	#print(ci)
	#ci now sum of descriptors
			
	ci=np.array(ci)	
	centroid=ci/c
	w=c/k
	l1=[]
	centroid=centroid.tolist()
	#print(centroid)
	l1.append(centroid)
	l1.append(w)		
	signatures.append(l1)
		
    print(signatures)

#    signatures=[[[1,2,3,4],2],[[2,4,5,6,7,8,9],3]]
  #  with open('outfile', 'wb') as fp:
 #   	pickle.dump(signatures, fp)






















    
