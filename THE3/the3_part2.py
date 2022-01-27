# -*- coding: utf-8 -*-
import cv2, math   # cv2.__version__ = '4.0.0'
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth #sklearn.__version__ :  0.24.2
from sklearn.datasets import make_blobs
from skimage import data, segmentation, color #skimage.__version__ :  0.17.2
from skimage.future import graph

__author__ = "Can Karagedik"
__student_id__ = "2288397"

def calculateHistogram(image, name = "x"):
    bins = list(range(256))
    histogram = [0]*256
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            histogram[image[i][j]] += 1
    fig1, axs1 = plt.subplots( nrows=1, ncols=1 )  
    fig1.set_size_inches(9, 7, forward=True)
    axs1.set_title("Image Histogram of {}".format(name))
    axs1.set_xlabel("intensity values")
    axs1.set_ylabel("number of pixels") 
    axs1.set_xlim(0, 255)
    axs1.grid(True)
    axs1.bar(bins, histogram, color = 'b')
    fig1.savefig(name + "_histogram.png")
    #fig1.show()

"""
def segmentation_function_VITAshift(image): # NOT USED, WRITTEN ONLY FOR UNDERSTANDING THE MEAN SHIFT ALGORITHM
    np.set_printoptions(suppress=True)
    number_of_rows = image.shape[0]
    number_of_columns = image.shape[1]
    segmented = np.zeros((number_of_rows, number_of_columns), np.uint8)
    print("gcd({},{}) = {}".format(number_of_rows, number_of_columns, math.gcd(number_of_rows, number_of_columns)))
    window_size = math.gcd(number_of_rows, number_of_columns) #bandwidth
    # Calculate initial means for every window
    means = np.empty((number_of_rows*number_of_columns//(window_size**2)), dtype=np.uint8)

    # All data points splited into as windows a window_size x window_size blocks
    # Then we calculate the centroids, the mean of each windows
    for x in range(number_of_rows//window_size):
        for y in range(number_of_columns//window_size):
            windows = image[window_size*x:window_size*(x + 1) , window_size*y:window_size*(y + 1)]
            windows = windows.flatten() 
            centroid = np.sum(windows)*(1/window_size**2)   # np.average(windows)
            np.append(means,int(centroid))
            #print("mean of the window {}-{} is {}".format((window_size*x,window_size*(x + 1)),(window_size*y,window_size*(y + 1)),int(centroid)))
    # After we cluster all means to a couple of means by sorting and taking average of window_size many of them.
    while (len(means) > window_size):
        print(len(means))
        print("means: ",means)
        means = np.sort(means, axis=None) 
        new_means = []
        for i in range (0,len(means)//(window_size)):
            mean = np.sum(means[i*(window_size):(i+1)*(window_size)])*(1/(window_size))
            new_means.append(int(mean))
            #print( new_means)
        #means = np.concatenate(means, new_means)
        new_means = np.array(new_means)
        means= new_means
    # Coloring the segments
    for i in range (0,len(means)):
        print("segment ",i," colored with",i*(255//(len(means)-1)))
        segment_row, segment_column = np.where( (image <= means[i+1] ) & (image >= means[i]))
        segmented[segment_row, segment_column] = i*(255//(len(means)-1))
    
    cv2.imshow("Segments",segmented)
"""    
def enhance(image, brightness = 127, contrast = 127 ):
    image = np.int16(image)
    enhanced = image * (contrast/127) + brightness - contrast
    normalized = np.clip(enhanced, 0, 255)
    enhanced = np.uint8(normalized)
    return enhanced

def segmentation_function_meanshift(image,quantile):
    shape = image.shape
    result = np.zeros((shape[0], shape[1]), np.uint8)
    reshape_img = np.reshape(image, [-1, 3])
    # Calculate gaussian kernel that is the bandwidth or a search window size
    # Quantile should be between [0, 1] (0.5 means that the median of all pairwise distances is used).
    # N_samples is the number of samples to use. If not given, all samples are used.
    bandwidth = estimate_bandwidth(reshape_img, quantile= quantile, n_samples=500) #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.estimate_bandwidth.html#sklearn.cluster.estimate_bandwidth
    #print("bandwidth: ",bandwidth)
    clusters = MeanShift(bandwidth=bandwidth, cluster_all=True, bin_seeding=True) #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    clusters.fit(reshape_img)
    labels = clusters.labels_
    #print("labels: ",labels)
    cluster_centers = clusters.cluster_centers_
    #print("cluster_centers: ",cluster_centers)
    labels_unique = np.unique(labels)
    number_of_clusters = len(labels_unique)
    #print("number of estimated clusters : %d" % number_of_clusters)
    segmented = np.reshape(labels, shape[:2])
    if len(labels_unique) == 1:
        #print("only 1 segment")
        pass
    else:
        for i in range (0,len(labels_unique)):
            segment_row, segment_column = np.where(segmented == i)
            result[segment_row, segment_column] = i*255//(len(labels_unique)-1)

    return result

def segmentation_function_ncut(image, initial_number_of_segments = 400): # https://scikit-image.org/docs/0.19.x/auto_examples/segmentation/plot_ncut.html?highlight=normalized
    # compactness balances color proximity and space proximity. Higher values give more weight to space proximity, making superpixel shapes more square/cubic. 
    labels = segmentation.slic(image, compactness = 30, n_segments = initial_number_of_segments, start_label = 1)# https://scikit-image.org/docs/0.19.x/api/skimage.segmentation.html#skimage.segmentation.slic
    graph_similarity_matrix = graph.rag_mean_color(image, labels, mode='similarity')
    labels = graph.cut_normalized(labels, graph_similarity_matrix)
    result = color.label2rgb(labels, image, bg_label = 0, kind = 'avg')
    return result


mean_contrast = [127,75,75]
ncut_contrast = [127,127,127] 
mean_parameters = [0.45,0.1,0.13] #[0.4,0.1,0.2] # Quantile values for estimate_bandwidth()
ncut_parameters = [200,50,50] #[300,50,50] # initial_number_of_segments for segmentation.slic()
np.set_printoptions(suppress=True)  #If True, always print floating point numbers using fixed point notation. If False, then scientific notation is used.
for i in range (1,4):
    # Read the input image
    img = cv2.imread("./partB/B{}.png".format(i), 1) #0(cv2.IMREAD_GRAYSCALE) stands for GRAYSCALE
    enhanced_img_1 = enhance(img, 127, mean_contrast[i-1])
    enhanced_img_2 = enhance(img, 127, ncut_contrast[i-1])
    #calculateHistogram(enhanced_gray, name = "B"+str(i)+"contrast75")
    #cv2.imshow("B"+str(i), img)
    #cv2.imshow("B"+str(i)+"enhanced", enhanced_img)
    mean_segmented = segmentation_function_meanshift(enhanced_img_1, mean_parameters[i-1])
    ncut_segmented = segmentation_function_ncut(enhanced_img_2, ncut_parameters[i-1])
    #cv2.imshow("B"+str(i)+" mean segmented", mean_segmented)
    cv2.imwrite("the3_B{}output_meanshift.png".format(i), mean_segmented)
    #cv2.imshow("B"+str(i)+" ncut segmented", ncut_segmented)
    cv2.imwrite("the3_B{}output_ncut.png".format(i), ncut_segmented)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
