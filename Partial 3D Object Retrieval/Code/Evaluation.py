from os.path import exists, isdir, basename, join, splitext
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
from pickle import dump, HIGHEST_PROTOCOL
import argparse
import numpy as np
from numpy import *
import pickle
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
from iteration_utilities import deepflatten
from scipy.stats import wasserstein_distance
import numpy, scipy.io
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

EXTENSIONS = [".txt"]
PRE_ALLOCATION_BUFFER = 1000  
GMM_FILE = 'gmm.file'


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covars_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

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

def normalize(fisher_vector):
	v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
	return v / np.sqrt(np.dot(v, v))

if __name__ == '__main__':
    print("---------------------")
    print("## loading feature files")
    datasetpath = "features/try"  
    all_files = []
    all_features = {}

    cat_files = get_files(datasetpath)
    cat_features = read_file(cat_files)
    all_files = all_files + cat_files
    all_features.update(cat_features)
    '''print(cat_files)
    print(cat_features)
    print(all_files)
    print(all_features)'''
    
    print("---------------------")
    all_features_array = dict2numpy(all_features)
    nfeatures = all_features_array.shape[0]

    #print("All features: \n", all_features_array)
    print("No of features %d" % int(nfeatures))

    
    N = int(nfeatures)
    K = 10

    gmm = GMM(n_components=K, covariance_type='diag')
    gmm.fit(all_features_array)

    #pickle.dump(gmm, open('gmm.model', 'wb'))

    print("---------------------")
    print("Compute the fisher vector encoding")
    all_fv = {}
    for i in all_features:
        fv = fisher_vector(all_features[i], gmm)
        fv = normalize(fv)
        all_fv[i] = fv

    #print(l2)

    l2 = []
    for i in range(len(all_features)+1):
        l2.append(0)

    for i in all_features:
        a, b = i.split('.')
        c, d, e = a.split('/')
        f = e[2:6]
        g = f.lstrip("0")
        l1 = []
        for j in range(len(all_fv[i])):
            l1.append(all_fv[i][j])
        l2[(int(g))] = l1

    #print(l2)
    print("L2:", l2[2])
    print("FV:", all_fv["features/try/D00002.txt"])

    print("---------------------")
    print("Compute the distance")

    #Euclidean Distance
    dist = []
    for i in range(1,len(l2)):
        q = []
        q.append(l2[i])
        d = euclidean_distances(l2[1:], q)
        dist.append(d)

    #EMD
    '''dist = []
    for i in range(1,len(l2)):
        for j in range(1, len(l2)):        
            d = wasserstein_distance(l2[j],l2[i])
            dist.append(d)'''

    #Chi_Square Distance
    '''dist = []
    for i in range(1,len(l2)):
        for j in range(1, len(l2)):        
            d = 0
            for k in range(1, len(l2[j])):
                sum1 = l2[j][k] + l2[i][k]	
                diff = l2[j][k] - l2[i][k]
                if( (sum1 > 0) and (diff != 0)):
                    d = d + (diff*diff)/sum1;
            dist.append((1/2)*d)'''

    #Cosine_Similarity
    '''dist = [] 
    for i in range(1,len(l2)):
        q = []
        q.append(l2[i])
        d = cosine_similarity(l2[1:], q)
        dist.append(d)'''

    #Cosine_Distance
    '''dist = []
    for i in range(1,len(l2)):
        for j in range(1, len(l2)):        
            d = spatial.distance.cosine(l2[j],l2[i])
            dist.append(d)'''  
   
    #print(dist)
    #print(dist[4][4])
    print("Distance Computation done!!!")
    print("---------------------")    

    print("Writing the dissimalrity matrix to a file...")
    dist_file = open("distance.txt", "w")

    #Euclidean Distance
    for i in range(len(dist)):
        for j in range(len(dist)):
            dist_file.write("%f " % dist[i][j][0])
        dist_file.write("\n")

    #EMD
    '''for i in range(len(dist)):
        dist_file.write("%f " % dist[i])
        if ((i+1) % (len(l2)-1) == 0):
            dist_file.write("\n")'''

    #Chi_Square Distance
    '''for i in range(len(dist)):
        dist_file.write("%f " % dist[i])
        if ((i+1) % (len(l2)-1) == 0):
            dist_file.write("\n")'''

    #Cosine_Distance
    '''for i in range(len(dist)):
        dist_file.write("%f " % dist[i])
        if ((i+1) % (len(l2)-1) == 0):
            dist_file.write("\n")'''

    print("Dissimilarity matrix successfully written to a file")
