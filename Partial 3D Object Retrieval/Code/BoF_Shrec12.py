from os.path import exists, isdir, basename, join, splitext
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
from pickle import dump, HIGHEST_PROTOCOL
import argparse
import numpy as np
from numpy import *
import pickle
import sys
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import euclidean_distances
from iteration_utilities import deepflatten
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

EXTENSIONS = [".txt"]
DATASETPATH = '../dataset'
PRE_ALLOCATION_BUFFER = 1000  
HISTOGRAMS_FILE = 'trainingdata.svm'
K_THRESH = 1  
CODEBOOK_FILE = 'codebook.file'

dimension = sys.argv[1]
dimension = int(dimension)

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
    #print(filename)
    f = open(filename, 'r')
    num = len(f.readlines())          # the number of features       
    featlength = dimension            # the length of the descriptor
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
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, dimension))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, dimension))
    return array

def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words

def normalize(fisher_vector):
	v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
	return v / np.sqrt(np.dot(v, v))

if __name__ == '__main__':
    print("---------------------")
    datasetpath = sys.argv[2] 
    print("## loading feature files from " + datasetpath)
    all_files = []
    all_features = {}

    cat_files = get_files(datasetpath)
    cat_features = read_file(cat_files)
    all_files = all_files + cat_files
    all_features.update(cat_features)
    
    print("---------------------")
    all_features_array = dict2numpy(all_features)
    nfeatures = all_features_array.shape[0]

    n = sys.argv[3]
    nclusters = int(n)
    print("No of features : %d" % nfeatures)
    print("No of clusters : %d" % int(nclusters))

    codebook, distortion = vq.kmeans(all_features_array,
                                             nclusters,
                                             thresh=K_THRESH)

    print("---------------------")
    print("Compute the visual words histograms for each object")
    all_word_histograms = {}
    for i in all_features:
        word_histogram = computeHistograms(codebook, all_features[i])
        all_word_histograms[i] = word_histogram

    l2 = []
    for i in range(len(all_features)+1):
        l2.append(0)

    for i in all_features:
        a1, b1 = i.split('..')
        a, b = b1.split('.')
        c, d, e, f1 = a.split('/')
        f = f1[2:6]
        g = f.lstrip("0")
        l1 = []
        for j in range(len(all_word_histograms[i])):
            l1.append(all_word_histograms[i][j])
        l2[(int(g))] = l1


    print("---------------------")

    with open("encoding.txt", "wb") as fp:   
        pickle.dump(l2, fp)
