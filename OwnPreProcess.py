import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.cluster import KMeans

#knearst process for missing data k=1
def knearst_process_vals(vals):
    process_index=0;
    valid_len = 0
    for line in vals:
        print process_index
        if np.isnan(np.sum(line)):
            if valid_len == 0:
                for i in range(0,len(line)):
                    if np.isnan(line[i]):
                        valid_len = i
                        break
            nearst_line = [];
            min_dist = 999999.0;
            for tmp_line in vals:
                if not np.isnan(np.sum(tmp_line)):
                    tmp_dist = abs(np.sum(np.array(line[:valid_len])-np.array(tmp_line[:valid_len])))
                    if(tmp_dist<min_dist):
                        nearst_line = tmp_line
                        min_dist = tmp_dist
            line[valid_len:] = nearst_line[valid_len:]
        process_index = process_index + 1
    return vals

#kmeans process for missing data
def Kmeans_process_vals(vals):
    cluster_list = []
    nan_list_index = []
    for i in range(0,len(vals)):
        if np.isnan(np.sum(vals[i])):
            nan_list_index.append(i)
        else:
            cluster_list.append(vals[i])
    print "start kmeans"
    print 'nan_list.size: ' + str(len(nan_list_index))
    print 'cluster_list.size: ' + str(len(cluster_list))
    cluster_num = 2 
    clf_kmeans = KMeans(n_clusters=cluster_num)
    s = clf_kmeans.fit(cluster_list)
    centers = clf_kmeans.cluster_centers_
    print "finish kemans"
    #process nan data
    process_index=0;
    for i in nan_list_index:
        nan_indexs = np.array(np.where(np.isnan(vals[i]))[0])
        no_nan_indexs = np.array(list(set(range(len(vals[i]))) -  set(nan_indexs)))
        vals[i][nan_indexs] = centers[0][nan_indexs]
        min_dist = abs(np.sum(np.array(centers[0])[no_nan_indexs]-vals[i][no_nan_indexs]))
        for j in range(1,cluster_num):
            tmp_dist = abs(np.sum(np.array(centers[j])[no_nan_indexs]-vals[i][no_nan_indexs]))
            if tmp_dist<=min_dist:
                min_dist = tmp_dist
                vals[i][nan_indexs] = centers[j][nan_indexs]
        process_index = process_index + 1
        #print 'process staus: ' + str(process_index) + '/' + str(len(nan_list_index))
    return vals
def Kmeans_process_vals_norm(vals):
    cluster_list = []
    nan_list_index = []
    for i in range(0,len(vals)):
        if np.isnan(np.sum(vals[i])):
            nan_list_index.append(i)
        else:
            cluster_list.append(vals[i])
    print "start kmeans"
    cluster_num = 4 
    stand_scale = preprocessing.MinMaxScaler()
    tmp_norm_vals = stand_scale.fit_transform(np.array(cluster_list))
    clf_kmeans = KMeans(n_clusters=cluster_num)
    s = clf_kmeans.fit(tmp_norm_vals)
    centers = clf_kmeans.cluster_centers_
    source_centers = stand_scale.inverse_transform(centers)
    print "finish kemans"
    #process nan data
    process_index=0;
    for i in nan_list_index:
        nan_indexs = np.array(np.where(np.isnan(vals[i]))[0])
        no_nan_indexs = np.array(list(set(range(len(vals[i]))) -  set(nan_indexs)))
        vals[i][nan_indexs] = source_centers[0][nan_indexs]
        min_dist = abs(np.sum(np.array(centers[0])[no_nan_indexs]-stand_scale.fit_transform(vals[i])[no_nan_indexs]))
        for j in range(1,cluster_num):
            tmp_dist = abs(np.sum(np.array(centers[j])[no_nan_indexs]-stand_scale.fit_transform(vals[i])[no_nan_indexs]))
            if tmp_dist<=min_dist:
                min_dist = tmp_dist
                vals[i][nan_indexs] = source_centers[j][nan_indexs]
        process_index = process_index + 1
        #print 'process staus: ' + str(process_index) + '/' + str(len(nan_list_index))
    return vals
