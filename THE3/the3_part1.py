import cv2
import numpy as np
from matplotlib import pyplot as plt
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
    axs1.set_title("Image Histogram")
    axs1.set_xlabel("intensity values")
    axs1.set_ylabel("number of pixels") 
    axs1.set_xlim(0, 255)
    axs1.grid(True)
    axs1.bar(bins, histogram, color = 'b')
    fig1.savefig(name + "_histogram.png")
    #fig1.show()

def morphology(image, kernel, operation = "erote"):
    k_x, k_y = kernel.shape
    number_of_rows = image.shape[0]
    number_of_columns = image.shape[1]
    result = np.zeros((number_of_rows, number_of_columns),np.uint8 )
    for x in range ((k_x-1)//2,number_of_rows-(k_x-1)//2):
        for y in range ((k_y-1)//2,number_of_columns-(k_y-1)//2):
            img_block = image[x-(k_x-1)//2: x+1+(k_x-1)//2 ,y-(k_y-1)//2: y+1+(k_y-1)//2]
            #print(img_block.shape)
            temp = img_block*kernel
            if(operation == "erote"):
                result[x][y] =  np.min(temp)
            elif (operation == "dilate"):
                result[x][y] = np.max(temp)
                
    return result

np.set_printoptions(suppress=True)  #If True, always print floating point numbers using fixed point notation. If False, then scientific notation is used.    
thresholds = [50,100,160,50,80]   
structuring_elements_erotion = [(3,3),(5, 5),(9, 9),(5, 5),(5, 5)] 
structuring_elements_dilation = [(5, 5),(5, 5),(3, 3),(5, 5),(7, 7)]
for i in range (1,6):
    # Read the input images in order
    img = cv2.imread("./partA/A{}.png".format(i), 1) #0(cv2.IMREAD_GRAYSCALE) stands for GRAYSCALE
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv2.imshow("A"+str(i), img)
    number_of_rows = img.shape[0]
    number_of_columns = img.shape[1]
    ret,binary = cv2.threshold(gray,thresholds[i-1],255,cv2.THRESH_BINARY_INV) #cv2.THRESH_BINARY_INV
    #calculateHistogram(gray, name = "A"+str(i))
    if (i == 3): #for the third image we apply opening instead of closing
        structuring_element = np.ones(structuring_elements_dilation[i-1], np.uint8 )
        dilated = morphology(binary,structuring_element,"dilate")
        structuring_element = np.ones(structuring_elements_erotion[2], np.uint8 )
        eroted = morphology(dilated,structuring_element,"erote")
        detected = eroted 
    else:
        structuring_element = np.ones(structuring_elements_erotion[i-1], np.uint8 )
        eroted = morphology(binary,structuring_element,"erote")
        structuring_element = np.ones(structuring_elements_dilation[i-1], np.uint8 )
        dilated = morphology(eroted,structuring_element,"dilate")
        detected = dilated
    
    contours, hierarchy = cv2.findContours(detected , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("The number of flying balloons in image A{} is {}".format(i,len(contours)-1))
    cv2.imwrite("part1_A{}.png".format(i), detected)
    
    # FOR THE NUMBERED OUTPUT FOR THE BALLOONS PLEASE UNCOMMENT
    """
    # Sort all the contours wrt their areas to discard ground
    objects = sorted(contours, key=cv2.contourArea)
    objects.pop(-1)
    # Get the moments https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
    moments = [None]*len(objects)
    for j in range(len(objects)):
        moments[j] = cv2.moments(objects[j])
    # Get the mass centers
    centers = [None]*len(objects)
    for j in range(len(objects)):
        # add 1e-5 to avoid division by zero
        centers[j] = (moments[j]['m10'] / (moments[j]['m00'] + 1e-5), moments[j]['m01'] / (moments[j]['m00'] + 1e-5)) 
    for j in range(len(objects)):
        color = (0, 0, 255)
        cv2.circle(img, (int(centers[j][0]), int(centers[j][1])), 1, color, -1)
        cv2.putText(img, str(j+1), (int(centers[j][0]) , int(centers[j][1]) + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    result = cv2.drawContours(img, objects, -1, (0,255,0), 3)
    cv2.imwrite("part1_A{}.png".format(i), result)
    """
    
