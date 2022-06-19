import cv2, random
import numpy as np
from matplotlib import pyplot as plt

path = "C:/Users/asus/Desktop/ComputerVision/Image Segmentation K-means/"
np.set_printoptions(suppress=True)  #If True, always print floating point numbers using fixed point notation. If False, then scientific notation is used.

def calculateHistogram(image, cluster_centers, name = "x"):
    bins = list(range(256))
    histogram = [0]*256
    max = 0
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            histogram[image[i][j]] += 1
            if histogram[image[i][j]] > max:
                max = histogram[image[i][j]]
            
    fig1, axs1 = plt.subplots( nrows=1, ncols=1 )  
    fig1.set_size_inches(9, 7, forward=True)
    axs1.set_title("GrayScale Image Histogram")
    axs1.set_xlabel("intensity values")
    axs1.set_ylabel("number of pixels") 
    axs1.set_xlim(0, 255)
    axs1.grid(True)
    axs1.bar(bins, histogram, color = 'b')
    
    for point in cluster_centers:
        #print(point)
        axs1.plot(point, max, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
        axs1.annotate(str(point), (point, max))
    fig1.savefig(path + name + "_histogram.png")
    fig1.clf()
    
    
def calculateTotalPixelValues(image):
    total = 0
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            total+= abs(image[i][j])
    return total

def optimizedKmeans(image, input, previous_output, previous_difference, K):
    #print(input)
    # Kmeans criterias:
    max_iter = 10
    epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    # Apply Kmeans
    compactness, labels, centers = cv2.kmeans(input, K, None, criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    output = centers[labels.flatten()]
    output = output.reshape((image.shape))

    #Distance/color-based difference
    difference = cv2.subtract(output, previous_output)
    B_diff, G_diff, R_diff = cv2.split(difference)
    total_difference = calculateTotalPixelValues(B_diff) + calculateTotalPixelValues(G_diff) + calculateTotalPixelValues(R_diff)
    total_difference = total_difference/(image.shape[0]*image.shape[1])
    """
    print("total_difference: ",total_difference)
    print(f"abs(previous_difference/(total_difference - previous_difference)) = |{previous_difference}/({total_difference} - {previous_difference})| =",abs(previous_difference/(total_difference - previous_difference)))
    """
    if abs(previous_difference/(previous_difference - total_difference))> 10 and K>1:
        colors = ['b','g','r','c','m','y','k',"#07FA83","#937097","#13FF00","#FF9B00","#CB4759","#2E4053"]
        # Color each cluster in different color (up to 13 color)
        for cluster_number in range(0,K+1):
            color = colors[cluster_number%len(colors)]
            A = input[labels.ravel()== cluster_number]
            plt.scatter(A[:,0],A[:,1], c = color)
        plt.scatter(centers[:,0],centers[:,1],s = 80,c = (0.5, 0.0, 0.5, 0.5), marker = 's')
        for center in centers:
            plt.annotate(str(center), (center[0],center[1]), color = (0.6, 0.6, 0.6))
        plt.savefig(path+f"{i}_clustersK={str(K)}")
        #plt.show()
        plt.close()
        plt.clf()
        centers = np.uint8(centers)
        output = centers[labels.flatten()]
        output = output.reshape((image.shape))
        result = np.concatenate((image,output),axis = 1)
        
        print("K:",K)
        cv2.imwrite(path + f"{i}_result_K={str(K)}.jpg",result) # save segmented output

        cluster_centers = []
        for center in centers:
            #print(center.reshape((-1,3)))
            B,G,R = center
            #print(f"B: {center[0]} G: {center[1]} R: {center[2]}")
            # grayscale conversion algorithm that OpenCV’s cvtColor() use
            grayvalue = 0.299 * R + 0.587 * G + 0.114 * B
            cluster_centers.append(int(grayvalue))
        calculateHistogram(np.uint8(input), cluster_centers,str(i))
    else:
        K += 1
        optimizedKmeans(image,input,output,total_difference,K)



if __name__ == "__main__":
    print("...")
    for i in range(1,7):
        print("IMAGE: ",i)
        image = cv2.imread(path+f"{i}.jpg",1)
        #image = cv2.resize(image,(400,200))
        #cv2.imshow("original",image)
        one_dim = image.reshape((-1,3))
        one_dim = np.float32(one_dim)

        zero = np.zeros(image.shape, np.uint8)
        optimizedKmeans(image, one_dim, zero,1, K = 1)
        
        print(f"IMAGE {i} finished")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
