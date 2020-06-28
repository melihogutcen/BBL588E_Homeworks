# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:16:45 2020

@author: melihogutcen
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

img = cv.imread("SunnyLake.bmp")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
pixel_values = img.reshape((-1,3))
pixel_values = np.float32(pixel_values)
#%% Using scaled RGB image calculation of Elbow and Silhouette Coeff
# The algorithm may take time. You can skip this part, the results are shown in report.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pixel_values)
wcss = []
identified_cluster = []
silhouette_scores = []

for i in range(2,10):
    kmeans = KMeans(i)
    kmeans.fit(scaled_data)
    identified_cluster=kmeans.fit_predict(scaled_data)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
    silhouette_scores.append(silhouette_score(scaled_data,kmeans.labels_,metric="euclidean"))

plt.figure(1) 
plt.plot(range(2,10),wcss,"o-")
plt.grid()
plt.title('Elbow Method')
plt.ylabel("Inertia or Distortion")
plt.xlabel("Cluster number")
plt.figure(2)
plt.plot(range(2,10),silhouette_scores,"o-")
plt.title("Silhouette Method")
plt.ylabel("silhouette score")
plt.xlabel("Cluster number")
plt.grid()
plt.show()
#%% using RGB image segmentation with cv.KMeans
# stopping criteria
criteria = (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,100,0.2)
fig, axs = plt.subplots(1,5,sharey=True,figsize=(10,8))
ctr,k = 0,2
for ax in axs:
    _, labels, (centers) = cv.kmeans(pixel_values, k, None,criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(img.shape)
    ax.imshow(segmented_img,interpolation='none')
    ax.set_title(["Cluster=",k],fontsize=8)
    ctr+=1
    k+=2
plt.show()
#%% Mean Shift Clustering Based Segmentation
originImg = cv.imread('SunnyLake.bmp')
originShape = originImg.shape   
flatImg=np.reshape(originImg, [-1, 3])    
bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True) 
ms.fit(flatImg)  
labels=ms.labels_   
cluster_centers = ms.cluster_centers_      
labels_unique = np.unique(labels)    
n_clusters_ = len(labels_unique)    
print("number of estimated clusters : %d" % n_clusters_)
segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
cv.imshow("Segmented Image",segmentedImg.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()