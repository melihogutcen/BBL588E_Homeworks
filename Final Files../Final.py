# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 02:00:44 2020

@author: melihogutcen
"""
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn import metrics
from sklearn.metrics import jaccard_similarity_score


# load the image and labelling
label = scipy.io.loadmat('image_label2.mat')
label = label['label'].tolist()[0] # ROI of bee
image = img_as_float(plt.imread("ITU_Golet_Ari.jpeg"))
image = image[label[1]:label[1]+1+label[3],label[0]:label[0]+label[2]+1,:]
# loop over the number of segments
for numSegments in (3, 9, 10):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
	segments = slic(image, n_segments = numSegments, sigma = 5)
	# show the output of SLIC
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))
	plt.axis("off")

plt.show()
plt.imshow(segments)
# The center of superpixels belongs to the bee.
row,col = segments.shape
bee_index = segments[round(row/2),round(col/2)]
bee_region = (segments==bee_index)*1
plt.imshow(bee_region)
plt.title("Segmented Bee Region")
#%% Evaluation

groundtruth_img = scipy.io.loadmat("maxarea.mat")
groundtruth_img = groundtruth_img["maxarea"]
segmented_img = bee_region
flat_groundtruth = groundtruth_img.ravel()
flat_segmented = segmented_img.ravel()

[tn, fp], [fn, tp] = metrics.confusion_matrix(flat_groundtruth,flat_segmented)
tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)
print("Confusion Matrix:\n",tp,fn,"\n",fp,tn)
dice = np.sum(flat_groundtruth[flat_segmented==1])*2.0 / (np.sum(flat_segmented) + np.sum(flat_groundtruth))
iou = jaccard_similarity_score(flat_groundtruth,flat_segmented)
print("Intersection over union: ",iou)
#%%  Histogram based approach
from skimage import exposure
from skimage import filters
val = filters.threshold_otsu(gray)
gray = 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]
hist, bins_center = exposure.histogram(gray)
plt.imshow(gray < val, cmap='gray', interpolation='nearest')
plt.axis('off')
#%%


