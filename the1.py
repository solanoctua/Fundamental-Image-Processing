import cv2, math, os
import numpy as np
from matplotlib import pyplot as plt

__author__ = "Can Karagedik"
__student_id__ = "2288397"


def CumulativeHistogram(histogram):
    cumulative_histogram  = [0]*256
    cumulative = 0
    i = 0
   
    for pixelvalue in histogram:
        
        cumulative += pixelvalue
        cumulative_histogram[i] = cumulative
        i += 1
    return cumulative_histogram

def HistogramEqualization(cumulative_histogram, image_size):
    equalized_histogram  = [0]*256
    i = 0
    for pixelvalue in cumulative_histogram:
       equalized_histogram[i] = round(((256-1)/image_size)*pixelvalue)
       i += 1
    return equalized_histogram

def GaussianDistribution(x, mean, standart_deviation):
    
    return (1/(standart_deviation*math.sqrt(2*math.pi)))*math.exp((-1/2)*pow((x - mean)/standart_deviation,2))
    
    
def part1(input_img_path, output_path, m, s):  #m,s
    # Read the input image
    img = cv2.imread(input_img_path, 0) #0(cv2.IMREAD_GRAYSCALE) stands for GRAYSCALE 
    img_size = img.shape[0]*img.shape[1] # number of pixels in the image
    
    # Create the output path if it does not exist.
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print("Directory ", output_path,  " Created")
    else:    
        print("Directory ", output_path, " already exists")

   
    histogram  = [0]*256
    
    
    last = [0]*256
    
    # Extract the histogram of the given image.
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            #print("intensity value = ", img[i][j])
            histogram[img[i][j]] += 1
    
    # Generate the histogram using Mixture of Gaussians with the given means and deviations.
    bins = list(range(256))
    gaussian_histogram_1  = [0]*256
    gaussian_histogram_2  = [0]*256
    gaussian_mixture = [0]*256
    for x in bins:
        gaussian_histogram_1[x] = GaussianDistribution(x, m[0], s[0])*img_size
        gaussian_histogram_2[x] = GaussianDistribution(x, m[1], s[1])*img_size
        gaussian_mixture[x] = (gaussian_histogram_1[x] + gaussian_histogram_2[x])
       
    
    # Histogram Equalization
    histogram_cumulative = CumulativeHistogram(histogram)
    histogram_equalized = HistogramEqualization(histogram_cumulative,img_size)

    histogram_gaussian_cumulative = CumulativeHistogram(gaussian_mixture)
    # Note that the sums of cumulative histograms are way different,
    # We need to make them equal except the distribution of pixel values.
    # So we multiply desired cumulative histogram with the ratio of sums
    
    ratio = histogram_cumulative[255] / histogram_gaussian_cumulative[255]
    #print("ratio: ", ratio)
    for i in range(0,256):
        histogram_gaussian_cumulative[i] = histogram_gaussian_cumulative[i] * ratio
        
    histogram_gaussian_equalized = [0]*256
    histogram_gaussian_equalized = HistogramEqualization(histogram_gaussian_cumulative,img_size)
    
    # Histogram Matching (or histogram stretching or histogram specification)
    # Determine the look-up table which represents our transformation function sending original image pixel values to desired pixel values
    mapping = [0]*256
    for i in range (256):
       mapping[i]= histogram_gaussian_equalized[histogram_equalized[i]]    
    
    processed_image = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    matched_image_histogram = [0]*256
    for i in range(0, img.shape[0]):
        
        for j in range(0, img.shape[1]):
            #print("intensity = ", img[i][j], "changed to ", atlast[img[i][j]])
            processed_image[i][j] = mapping[img[i][j]]
            matched_image_histogram[mapping[img[i][j]]] += 1
            

    fig1, axs1 = plt.subplots( nrows=1, ncols=1 )  
    fig1.set_size_inches(9, 7, forward=True)
    axs1.set_title("Original Image Histogram")
    axs1.set_xlabel("intensity values")
    axs1.set_ylabel("number of pixels") 
    axs1.set_xlim(0, 255)
    axs1.grid(True)
    axs1.bar(bins, histogram, color = 'b')
    fig1.savefig(output_path + "original_histogram.png")
    #fig1.show()
    
    fig2, ax2 = plt.subplots( nrows=1, ncols=1 )
    fig2.set_size_inches(9, 7, forward=True)
    ax2.set_title("Matched Image Histogram")
    ax2.set_xlabel("intensity values")
    ax2.set_ylabel("number of pixels") 
    ax2.set_xlim(0, 255)
    ax2.grid(True)
    ax2.bar(bins, matched_image_histogram, color = 'b')
    fig2.savefig(output_path + "matched_image_histogram.png")
    #fig2.show()

    fig3, axs3 = plt.subplots( nrows=1, ncols=1 )  
    fig3.set_size_inches(9, 7, forward=True)
    axs3.set_title("Gaussian Histogram")
    axs3.set_xlabel("intensity values")
    axs3.set_ylabel("number of pixels") 
    axs3.set_xlim(0, 255)
    axs3.grid(True)
    axs3.bar(bins, gaussian_mixture, color = 'b')
    fig3.savefig(output_path + "gausssian_histogram.png")
    #fig3.show()
    
    cv2.imwrite(output_path + "matched_image.png", processed_image)
    return processed_image
    

def the1_convolution(input_img_path, filter):
    img = cv2.imread(input_img_path)
    
    m, n = filter.shape
    if (m == n): # check if the given filter(kernel) is square or not
        number_of_rows = img.shape[0]
        number_of_columns = img.shape[1]
        
        # Create an empty RGB image, that is a  matrix equal the size of the input image and the entries are full of zeros for each R,G,B channel.
        processed_img = np.zeros((number_of_rows,number_of_columns,3), np.uint8)            
        # Loop over each channel(R,G,B) seperately for convolution
        for channel in range(img.shape[2]):
            # Loop over entries (y,x)
            for i in range((m-1)//2, number_of_rows - ((m-1)//2) ):  # Due to the nature of the convolution operation, endmost m-1 rows and m-1 columns will not be calculated
                for j in range((m-1)//2, number_of_columns - (m-1)//2 ):  # use // to get an integer result not float (4/2 = 2.0 where as 4//2 = 2)
                    processed_img[i][j][channel] =(1)*round(np.sum(img[i-((m-1)//2):i+1-((m-1)//2), j-((m-1)//2):j+1-((m-1)//2), channel]*filter))
                                    
    return processed_img

def part2(input_img_path, output_path):
    img = cv2.imread(input_img_path, 0) #0(cv2.IMREAD_GRAYSCALE) stands for GRAYSCALE 

    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print("Directory ", output_path,  " Created")
    else:    
        print("Directory ", output_path, " already exists")
     
    number_of_rows = img.shape[0]
    number_of_columns = img.shape[1]
    
    # CANNY EDGE DETECTION 
    # Smoothing(noise removing)
    img = cv2.GaussianBlur(img,(5,5),0)
    """

    GaussianBlur = np.array([[1,  4,  7,  4, 1],
                             [4, 16, 26, 16, 4],
                             [7, 26, 41, 26, 7],
                             [4, 16, 26, 16, 4],
                             [1,  4,  7,  4, 1]])
    
    GaussianBlur2 = np.array([[2,  4,  5,  4, 2],
                              [5,  9, 26,  9, 4],
                              [5, 12, 15, 12, 5],
                              [4,  9, 12,  9, 4],
                              [2,  4,  5,  4, 2]])
    filter =  GaussianBlur2
    m, n = filter.shape
    blured = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    for i in range((m-1)//2, number_of_rows - ((m-1)//2) ):  
                for j in range((m-1)//2, number_of_columns - (m-1)//2 ):  
                    blured[i][j] = (1/159)*round(np.sum(img[i-2:i+3, j-2:j+3]*filter))
    
    """
    # Compute gradients using Sobel Operator  https://en.wikipedia.org/wiki/Sobel_operator
    sobel_horizontal = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])
    
    sobel_vertical = np.array([[-1, -2, -1],
                               [ 0 , 0 , 0],
                               [ 1 , 2 , 1]])
    
    
    gradient_magnitudes = np.zeros((number_of_rows, number_of_columns), np.uint8)
    gradient_horizontal = np.zeros((number_of_rows, number_of_columns), np.uint8)
    gradient_vertical = np.zeros((number_of_rows, number_of_columns), np.uint8)
    
    # Since sobel kernel is 3x3, due to the nature of the convoluton operation endmost rows and columns will be omitted
    for i in range(1, number_of_rows - 1):
        for j in range(1, number_of_columns - 1):
            gradient_horizontal[i][j] = round(np.sum(img[i-1:i+2, j-1:j+2]*sobel_horizontal))
            gradient_vertical[i][j] = round(np.sum(img[i-1:i+2, j-1:j+2]*sobel_vertical))
            gradient_magnitudes[i][j] = math.sqrt(pow(round(np.sum(img[i-1:i+2, j-1:j+2]*sobel_horizontal)),2) + pow(round(np.sum(img[i-1:i+2, j-1:j+2]*sobel_vertical)),2))
           
    edge_directions = np.arctan2(gradient_horizontal, gradient_vertical)#four-quadrant inverse tangent
    edge_directions  = np.rad2deg(edge_directions ) # values between  (-180,180)
     
    # Non-maximum suppression (edge thinning technique) https://theailearner.com/tag/non-max-suppression/      http://www.adeveloperdiary.com/data-science/computer-vision/implement-canny-edge-detector-using-python-from-scratch/
    non_max = np.zeros((number_of_rows, number_of_columns), np.uint8)
    for i in range(1, number_of_rows - 1):
        for j in range(1, number_of_columns - 1):
            edge_direction = edge_directions[i, j]
            # We shall consider 8-neighborhood system
            # For each pixel, the neighboring pixels are located in horizontal, vertical, and diagonal directions (0°, 45°, 90°, and 135°).
            # After rounding, we will compare every pixel value against the two neighboring pixels in the gradient direction.
            # If that pixel is a local maximum, it is retained as an edge pixel otherwise suppressed. This way only the largest responses will be left.
            
            #East-West(right-left) direction
            if((0 <= abs(edge_direction) < 180/8) or (7*180/8 < abs(edge_direction) <= 180)):
                neighbor_1 = gradient_magnitudes[i, j + 1]
                neighbor_2 = gradient_magnitudes[i, j - 1]

            #SouthEast-NorthWest (bottom right-top left) direction
            elif((180/8 <= edge_direction < 3*180/8) or (-7*180/8 <= edge_direction < -5*180/8)):
                neighbor_1 = gradient_magnitudes[i + 1, j - 1]
                neighbor_2 = gradient_magnitudes[i - 1, j + 1]
            
            #North-South(up-down) direction 
            elif(3*180/8 <= abs(edge_direction) <= 5*180/8):
                neighbor_1 = gradient_magnitudes[i + 1, j ]
                neighbor_2 = gradient_magnitudes[i - 1, j ]
                
            #NorthEast-SouthWest (top right-bottom left) direction
            elif((5*180/8 < edge_direction <= 7*180/8) or (-3*180/8 <= edge_direction < 180/8)):
                neighbor_1 = gradient_magnitudes[i + 1, j + 1]
                neighbor_2 = gradient_magnitudes[i - 1, j - 1]
            
            if (gradient_magnitudes[i,j] >= neighbor_1 ) and (gradient_magnitudes[i,j] >= neighbor_2):
                non_max[i,j] = gradient_magnitudes[i,j]
            else:
                non_max[i,j] = 0
               
    # Thresholding  
    # If an edge pixel’s gradient value is higher than the high threshold value, it is marked as a strong edge pixel.
    # If an edge pixel’s gradient value is smaller than the high threshold value and larger than the low threshold value, it is marked as a weak edge pixel
    thresholded = np.zeros((number_of_rows, number_of_columns), np.uint8)
    threshold_low = 50
    threshold_high = 125
            
    strong_row, strong_column = np.where(non_max >= threshold_high )
    weak_row, weak_column = np.where( (non_max <= threshold_high ) & (non_max >= threshold_low))
    zero_row, zero_column = np.where(non_max < threshold_low)
            
    thresholded[zero_row, zero_column] = 0
    thresholded[weak_row, weak_column] = 100
    thresholded[strong_row, strong_column] = 255
     
    # Edge tracking by hysteresis
    # To track the edge connection, blob analysis is applied by looking at a weak edge pixel and its 8-connected neighborhood pixels.
    # As long as there is one strong edge pixel that is involved in the blob, that weak edge point can be identified as one that should be preserved.
    for i in range(1, number_of_rows - 1):
        for j in range(1, number_of_columns - 1):
            if (thresholded[i,j] == 100):
                if (255 in [thresholded[i-1:i+2, j-1:j+2].tolist()]):
                    thresholded[i, j] = 255
                else:
                    thresholded[i, j] = 0
             
    cv2.imwrite(output_path + "edges.png", thresholded)
    return thresholded

def enhance_3(path_to_3 , output_path):
    img = cv2.imread(path_to_3)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print("Directory ", output_path,  " Created")
    else:    
        print("Directory ", output_path, " already exists")
        
    
    (B, G, R) = cv2.split(img)
    
    average5 = np.ones((5,5),np.float32)/25
    B = cv2.filter2D(B,-1,average5)
    B = cv2.medianBlur(B,5)
    B = cv2.equalizeHist(B)
    
    G = cv2.filter2D(G,-1,average5)
    G = cv2.medianBlur(G,5)
    ret,G = cv2.threshold(G,115,255,cv2.THRESH_BINARY)
    
    R = cv2.filter2D(R,-1,average5)
    R = cv2.medianBlur(R,5)
    ret,R = cv2.threshold(R,112,255,cv2.THRESH_BINARY)
    
    enhanced = cv2.merge((B,R,G))
    cv2.imwrite(output_path + "enhanced1.png", enhanced)

    return enhanced

def enhance_4(path_to_4 , output_path):
    
    img = cv2.imread(path_to_4)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print("Directory ", output_path,  " Created")
    else:    
        print("Directory ", output_path, " already exists")
        
    (B, G, R) = cv2.split(img)
    
    average5 = np.ones((5,5),np.float32)/25
    average3 = np.ones((3,3),np.float32)/9
    sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    
    R = cv2.filter2D(R, -1, sharpen)
    R = cv2.filter2D(R,-1, average5)
    R = cv2.medianBlur(R,5)
    ret,R = cv2.threshold(R,115,255,cv2.THRESH_BINARY)
    
    B = cv2.filter2D(B, -1, sharpen)
    B = cv2.filter2D(B,-1, average3)

    G = cv2.filter2D(G, -1, sharpen)
    G = cv2.medianBlur(G,5)
    
    enhanced = cv2.merge((B,G,R))
    cv2.imwrite(output_path + "enhanced2.png",enhanced)
    
    return enhanced


np.set_printoptions(suppress=True)  #If True, always print floating point numbers using fixed point notation. If False, then scientific notation is used.
# Create the output path if it does not exist.

directory = os.getcwd()
print("Writing outputs to: ",directory)
print("please wait until you see \"finished\" text")
THE1_outputs_path = './THE1outputs/'
if not os.path.exists( THE1_outputs_path):
    os.mkdir(THE1_outputs_path)
    print("Directory ", THE1_outputs_path,  " Created")
else:    
    print("Directory ", THE1_outputs_path, " already exists")

THE1_outputs_path = './THE1outputs/Part1/'
if not os.path.exists( THE1_outputs_path):
    os.mkdir(THE1_outputs_path)
    print("Directory ", THE1_outputs_path,  " Created")
else:    
    print("Directory ", THE1_outputs_path, " already exists")
THE1_outputs_path = './THE1outputs/Part1/1/'
if not os.path.exists( THE1_outputs_path):
    os.mkdir(THE1_outputs_path)
    print("Directory ", THE1_outputs_path,  " Created")
else:    
    print("Directory ", THE1_outputs_path, " already exists")
    
THE1_outputs_path = './THE1outputs/Part1/2/'
if not os.path.exists( THE1_outputs_path):
    os.mkdir(THE1_outputs_path)
    print("Directory ", THE1_outputs_path,  " Created")
else:    
    print("Directory ", THE1_outputs_path, " already exists")
    
THE1_outputs_path = './THE1outputs/Part2/'
if not os.path.exists( THE1_outputs_path):
    os.mkdir(THE1_outputs_path)
    print("Directory ", THE1_outputs_path,  " Created")
else:    
    print("Directory ", THE1_outputs_path, " already exists")


part1('./THE1-Images/1.png','./THE1outputs/Part1/1/ex1/', [45,200], [45,45])
part1('./THE1-Images/1.png','./THE1outputs/Part1/1/ex2/', [50,200], [10,10])
part1('./THE1-Images/2.png','./THE1outputs/Part1/2/ex1/', [75,200], [40,40])
part1('./THE1-Images/2.png','./THE1outputs/Part1/2/ex2/', [75,200], [40,40])

the1_convolution('./THE1-Images/2.png', np.array([[1,1,1],[1,1,1],[1,1,1]]))

part2('./THE1-images/1.png','./THE1outputs/Part2/1/')
part2('./THE1-images/2.png','./THE1outputs/Part2/2/')

enhance_3('./THE1-images/3.png','./THE1outputs/Part2/3/')
enhance_4('./THE1-images/4.png','./THE1outputs/Part2/4/')
print("finished")

