import numpy as np
from iteration_utilities import deepflatten
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import sys

def read_file(filename):
              
    f = open(filename, 'r')
    num = len(f.readlines())        
    f.close()

    f = open(filename, 'r')                 
    dist = np.zeros((num, num));        
    data = []
    lines = f.readlines()

    for line in lines:
        data.append([float(v) for v in line.split()])

    for i in range(num):
        for j in range(num):
            dist[i, j] = data[i][j]

    f.close()
    return dist

if __name__ == '__main__':

    dist = read_file("PSBresult.matrix")

    #print(dist)

    total_objects = len(dist)
    each_class_objects = sys.argv[1]
    each_class_objects = int(each_class_objects)

    li = []
    for i in range(0, total_objects, each_class_objects):       
        li.append(i)

    avg_prec = []
    prec = []
    recall = []
    interpolated_prec = []
    interpolated_recall = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in range(len(dist)):
        k=1
        #Query Class
        for j in li:
            if(i >= j and i < j+each_class_objects):
                cla_query = k
                break
            k=k+1

        d = list(deepflatten(dist[i], depth=1))
        arr = np.array(d)
        l = 2*each_class_objects
        least = arr.argsort()[1:l]

    
        cla_ret = []
        for i in range(len(least)):
            k=1
            for j in li:
                if(least[i] >= j and least[i] < j+each_class_objects):
                    cla_ret.append(k)
                    k=k+1
                    break
                k=k+1


        relevant = each_class_objects
        retrieved = len(cla_ret)
        relevant_retrieved = 0
        for i in range(len(cla_ret)):
            if(cla_ret[i] == cla_query):
                relevant_retrieved = relevant_retrieved+1


        prec.append(relevant_retrieved/retrieved)
        recall.append(relevant_retrieved/relevant)

        a=0
        p = []
        r = []
        for i in range(len(cla_ret)):
            if(cla_ret[i] == cla_query):
                a=a+1  
            p.append(a/(i+1))
            r.append(a/relevant)

 
        sum_prec = []
        sum_recall = []
        for i in range(len(p)):
            if(cla_ret[i] == cla_query):
                sum_prec.append(p[i])
                sum_recall.append(r[i])
        
        in_pr = []
        for i in range(len(interpolated_recall)):
            precision1 = []
            for j in range(len(sum_recall)):
                if(sum_recall[j] >= interpolated_recall[i]):
                    precision1.append(sum_prec[j])
            if(len(precision1) == 0):
                in_pr.append(0)
            else:
                in_pr.append(max(precision1))


        interpolated_prec.append(in_pr)
    
        if(len(sum_prec) == 0):
            avg_prec.append(0)
        else:
            avg_prec.append(sum(sum_prec)/len(sum_prec))
    
    mAP = sum(avg_prec)/len(avg_prec)
    print("Mean Average Precision (mAP) = %.4f" % mAP)

    fileName = sys.argv[2]
    fp = open(fileName, "a")
    fp.write("\nmAP = %.4f\n" % mAP)

    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    p6 = []
    p7 = []
    p8 = []
    p9 = []
    p10 = []
    p11 = []

    for i in range(len(interpolated_prec)):
        p1.append(interpolated_prec[i][0])
        p2.append(interpolated_prec[i][1])
        p3.append(interpolated_prec[i][2])
        p4.append(interpolated_prec[i][3])
        p5.append(interpolated_prec[i][4])
        p6.append(interpolated_prec[i][5])
        p7.append(interpolated_prec[i][6])
        p8.append(interpolated_prec[i][7])
        p9.append(interpolated_prec[i][8])
        p10.append(interpolated_prec[i][9])
        p11.append(interpolated_prec[i][10])


    length = len(interpolated_prec)
    a1 = sum(p1) / length
    a2 = sum(p2) / length
    a3 = sum(p3) / length
    a4 = sum(p4) / length
    a5 = sum(p5) / length
    a6 = sum(p6) / length
    a7 = sum(p7) / length
    a8 = sum(p8) / length
    a9 = sum(p9) / length
    a10 = sum(p10) / length
    a11 = sum(p11) / length

    interpolated_precision = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]
    #print("Interpolated Precision: ", interpolated_precision)
    #print("Interpolated Recall: ", interpolated_recall)


    '''plt.plot(interpolated_recall, interpolated_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.ylim([0.0, 1.0])
    #plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: mAP={0:0.4f}'.format(mAP))
    plt.show()'''
