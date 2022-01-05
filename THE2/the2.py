import cv2, os, math, cmath 
import numpy as np
from pathlib import Path
__author__ = "Can Karagedik"
__student_id__ = "2288397"

np.set_printoptions(suppress=True)  #If True, always print floating point numbers using fixed point notation. If False, then scientific notation is used.
def create_output_path(output_path):
    output_path = Path(output_path)
    # Create the output paths if it do not exist.
    output_path.mkdir(parents=True, exist_ok=True)
    """
    if not os.path.exists( output_path):
        os.mkdir(output_path)
        
        print("Directory ", output_path,  " Created")
    else:    
        print("Directory ", output_path, " already exists")
    """

def DFT(image ):
    transformed = np.zeros(image.shape, np.uint8)
    number_of_rows = image.shape[0]
    number_of_columns = image.shape[1]
    for x in range(0, number_of_rows):
        for y in range(0, number_of_columns):
            summation = 0
            #print("F({},{}) = ".format(x,y))
            for u in range(0, number_of_rows):
                for v in range(0, number_of_columns):
                    #print(" {}*e^(2j*pi*(({}*{})/{} + ({}*{})/{}) +".format(image[x][y],x,u,number_of_rows, y,v,number_of_columns))
                    summation += image[x][y]*np.exp(-2j*np.pi*((x*u/number_of_rows) + (y*v/number_of_columns))) #summation += image[x][y]*math.exp(-2j*np.pi*(x*u/number_of_rows + y*v/number_of_columns))
            transformed[x][y] = summation   #(1/math.sqrt(number_of_rows*number_of_columns))*summation
    
def part1(input_img_path, output_path):
    # Read the input image
    img = cv2.imread(input_img_path, 0) #0(cv2.IMREAD_GRAYSCALE) stands for GRAYSCALE
    #img = cv2.resize(img, (480,480), interpolation= cv2.INTER_LINEAR)
    create_output_path(output_path)
    
    number_of_rows = img.shape[0]
    number_of_columns = img.shape[1]
    
    fourier = np.fft.fft2(img)
    # Concentrate high frequencies to the middle (right now they are at the corners)
    shifted = np.fft.fftshift(fourier) # Shift the zero-frequency component to the center of the spectrum.
    """
    magnitude = 20*np.log(np.abs(shifted)) #only for visualizing purposes 
    transformed = np.asarray(magnitude, np.uint8)
    """
    # Apply filters for edge detection
    
    Gaussian =  np.array([[2,  4,  5,  4, 2],
                         [4,  9, 12,  9, 4],
                         [5, 12, 15, 12, 5],
                         [4,  9, 12,  9, 4],
                         [2,  4,  5,  4, 2]])*(1/159)
    
    padding = (number_of_rows - 5, number_of_columns - 5)  # total amount of padding
    Gaussian = np.pad(Gaussian, (((padding[0]+1)//2, padding[0]//2), ((padding[1]+1)//2, padding[1]//2)), 'constant', constant_values = 1)
    Gaussian = np.fft.fft2(Gaussian)
    Gaussian = np.fft.fftshift(Gaussian)
    filtered = Gaussian * shifted
    
    sobel_combined = np.array([[-1, -2, 1],
                                 [-2, 0, 2],
                                 [-1, 2, 1]])

    padding = (number_of_rows - 3, number_of_columns - 3)  # total amount of padding
    sobel_combined = np.pad(sobel_combined, (((padding[0]+1)//2, padding[0]//2), ((padding[1]+1)//2, padding[1]//2)), 'constant')
    sobel_combined = np.fft.fft2(sobel_combined)
    sobel_combined = np.fft.fftshift(sobel_combined)
    filtered = filtered*sobel_combined

    # Apply inverse shift
    reshifted = np.fft.ifftshift(filtered)
    # Return to spatial domain using Inverse Fourier Transformation
    inversefourier = np.fft.ifft2(reshifted)
    inversefourier = np.abs(inversefourier)
    # Normalize the image
    inversefourier -= inversefourier.min()
    inversefourier = inversefourier*255 / inversefourier.max()
    spatial = inversefourier.astype(np.uint8)
    # Thresholding  
    # If an edge pixel’s gradient value is higher than the high threshold value, it is marked as a strong edge pixel.
    # If an edge pixel’s gradient value is smaller than the high threshold value and larger than the low threshold value, it is marked as a weak edge pixel
    thresholded = np.zeros((number_of_rows, number_of_columns), np.uint8)
    threshold_low = 30
    threshold_high = 60
            
    strong_row, strong_column = np.where(spatial >= threshold_high )
    weak_row, weak_column = np.where( (spatial <= threshold_high ) & (spatial >= threshold_low))
    zero_row, zero_column = np.where(spatial < threshold_low)
            
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
    
    #cv2.imshow("Edges", thresholded)
    cv2.imwrite(output_path + "edges.png", thresholded)
    return thresholded

def findPixelCoordinates(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x,y: ",x,y)

    return(x,y)
    
def passFilter(image, filtertype, radius = 100):
    number_of_rows = image.shape[0]
    number_of_columns = image.shape[1]
    
    highPass_filter = np.ones((number_of_rows, number_of_columns), np.uint8)
    #radius = 100  
    x_center, y_center = number_of_rows//2 ,number_of_columns//2
    x, y = np.ogrid[ :number_of_rows, :number_of_columns]
    if filtertype == "Highpass":
        circle_row, circle_column = np.where((x - x_center)**2 + (y - y_center )**2 <= radius**2 )
    elif filtertype == "Lowpass":
        circle_row, circle_column = np.where((x - x_center)**2 + (y - y_center )**2 >= radius**2 )
    highPass_filter[circle_row, circle_column] = 0

    return image * highPass_filter

def boxFilter(image, targetpixel, size):
    number_of_rows = image.shape[0]
    number_of_columns = image.shape[1]
    boxFilter = np.ones((number_of_rows, number_of_columns), np.uint8)
    boxFilter[targetpixel[0]-((size[0]-1)//2) : targetpixel[0]+((size[0]-1)//2),targetpixel[1]-((size[1]-1)//2) : targetpixel[1]+1+((size[1]-1)//2)] = 0
    return image * boxFilter
    
def bandRejectFilter(image, band = (20,40)):
    number_of_rows = image.shape[0]
    number_of_columns = image.shape[1]
    inner,outer = band[:2]
    #print(inner,outer)
    bandReject_filter = np.ones((number_of_rows, number_of_columns), np.uint8)
    
    x_center, y_center = number_of_rows//2 ,number_of_columns//2
    x, y = np.ogrid[ :number_of_rows, :number_of_columns]
    band_row, band_column = np.where(np.logical_and((x - x_center)**2 + (y - y_center )**2 <= outer**2 ,(x - x_center)**2 + (y - y_center )**2 >= inner**2 ))
    bandReject_filter[band_row, band_column] = 0
    return image * bandReject_filter

def enhance_3(path_to_3, output_path):
    # Read the input image
    img = cv2.imread(path_to_3)
    create_output_path(output_path)
    #img = cv2.resize(img, (480,480), interpolation= cv2.INTER_LINEAR)
    number_of_rows = img.shape[0]
    number_of_columns = img.shape[1]
    (B, G, R) = cv2.split(img)
    
    Bfourier = np.fft.fft2(B)
    Bshifted = np.fft.fftshift(Bfourier)# Shift the zero-frequency component to the center of the spectrum.
    Gfourier = np.fft.fft2(G)
    Gshifted = np.fft.fftshift(Gfourier)
    Rfourier = np.fft.fft2(R)
    Rshifted = np.fft.fftshift(Rfourier)
    # Click pixel and learn its pixel location
    #cv2.imshow("B Frequency Domain", Btransformed)
    #cv2.setMouseCallback("B Frequency Domain",findPixelCoordinates)
    
    # RED CHANNEL
    Rfiltered = passFilter(Rshifted,"Lowpass", 120)
    #Rfiltered = bandRejectFilter(Rfiltered, (14,18))
    Rfiltered = boxFilter(Rfiltered,(number_of_rows//2,number_of_columns//4),(5,number_of_rows//2 -30))  # left horizontal line
    Rfiltered = boxFilter(Rfiltered,(number_of_rows//4,number_of_columns//2),(number_of_columns//2 -15 ,5))  # upper vertical line
    Rfiltered = boxFilter(Rfiltered,(int(number_of_rows*3/4),number_of_columns//2),(number_of_columns//2 -15 ,5))  # bottom vertical line
    Rfiltered = boxFilter(Rfiltered,(number_of_rows//2,int(number_of_columns*3/4)),(5,number_of_rows//2 -30))  # right horizontal line
    # BLUE CHANNEL
    Bfiltered = passFilter(Bshifted,"Lowpass", 120)
    #Bfiltered = bandRejectFilter(Bfiltered, (14,18))
    Bfiltered = boxFilter(Bfiltered,(number_of_rows//2,number_of_columns//4),(4,number_of_rows//2 -30)) # left horizontal line
    Bfiltered = boxFilter(Bfiltered,(number_of_rows//4,number_of_columns//2),(number_of_columns//2 -15 ,2))  # upper vertical line
    Bfiltered = boxFilter(Bfiltered,(int(number_of_rows*3/4),number_of_columns//2),(number_of_columns//2 -15 ,2))  # bottom vertical line
    Bfiltered = boxFilter(Bfiltered,(number_of_rows//2,int(number_of_columns*3/4)),(4,number_of_rows//2 -30))  # right horizontal line
    # GREEN CHANNEL
    Gfiltered = passFilter(Gshifted,"Lowpass", 120)
    #Gfiltered = bandRejectFilter(Gfiltered, (14,18))
    Gfiltered = boxFilter(Gfiltered,(number_of_rows//2,number_of_columns//4),(4,number_of_rows//2 -30))  # left horizontal line
    Gfiltered = boxFilter(Gfiltered,(number_of_rows//4,number_of_columns//2),(number_of_columns//2 -15 ,2))  # upper vertical line
    Gfiltered = boxFilter(Gfiltered,(int(number_of_rows*3/4),number_of_columns//2),(number_of_columns//2 -15 ,2))  # bottom vertical line
    Gfiltered = boxFilter(Gfiltered,(number_of_rows//2,int(number_of_columns*3/4)),(4,number_of_rows//2 -30))  # right horizontal line
    """
    Bmagnitude = 20*np.log(np.abs(Bfiltered)) #only for visualizing purposes 
    Btransformed = np.asarray(Bmagnitude, np.uint8)
    Gmagnitude = 20*np.log(np.abs(Gfiltered)) #only for visualizing purposes 
    Gtransformed = np.asarray(Gmagnitude, np.uint8)
    Rmagnitude = 20*np.log(np.abs(Rfiltered)) #only for visualizing purposes 
    Rtransformed = np.asarray(Rmagnitude, np.uint8)
    filtered = np.concatenate((Btransformed,Gtransformed,Rtransformed),1)
    cv2.imshow("All Filtered", filtered)
    cv2.imwrite(output_path+"/3Filtered.png", filtered)
    """
    # Apply inverse shift
    Breshifted = np.fft.ifftshift(Bfiltered)
    Greshifted = np.fft.ifftshift(Gfiltered)
    Rreshifted = np.fft.ifftshift(Rfiltered)
    # Return to spatial domain using Inverse Fourier Transformation
    Binversefourier = np.fft.ifft2(Breshifted)
    Binversefourier = np.abs(Binversefourier)
    Ginversefourier = np.fft.ifft2(Greshifted)
    Ginversefourier = np.abs(Ginversefourier)
    Rinversefourier = np.fft.ifft2(Rreshifted)
    Rinversefourier = np.abs(Rinversefourier)
    # Normalize
    Binversefourier -= Binversefourier.min()
    Binversefourier = Binversefourier*255 / Binversefourier.max()
    Ginversefourier -= Ginversefourier.min()
    Ginversefourier = Ginversefourier*255 / Ginversefourier.max()
    Rinversefourier -= Rinversefourier.min()
    Rinversefourier = Rinversefourier*255 / Rinversefourier.max()
    
    Benhanced = Binversefourier.astype(np.uint8)
    Genhanced = Ginversefourier.astype(np.uint8)
    Renhanced = Rinversefourier.astype(np.uint8)
    enhanced = cv2.merge((Benhanced,Genhanced,Renhanced))
    #result = np.concatenate((Benhanced,Genhanced,Renhanced),1)
    #cv2.imshow("All Back to Spatial Domain", result)
    #v2.imwrite(output_path+"/3EnhancedB.png", Benhanced)
    #cv2.imwrite(output_path+"/3EnhancedG.png", Genhanced)
    #cv2.imwrite(output_path+"/3EnhancedR.png", Renhanced)
    #cv2.imwrite(output_path+"/3Enhanced.png", enhanced)
    cv2.imwrite(output_path+"/Enhanced.png", enhanced)
    return  enhanced

def enhance_4(path_to_4, output_path):
    # Read the input image
    img = cv2.imread(path_to_4)
    create_output_path(output_path)
    #img = cv2.resize(img, (480,480), interpolation= cv2.INTER_LINEAR)
    number_of_rows = img.shape[0]
    number_of_columns = img.shape[1]
    (B, G, R) = cv2.split(img)
    
    Bfourier = np.fft.fft2(B)
    Bshifted = np.fft.fftshift(Bfourier)# Shift the zero-frequency component to the center of the spectrum.
    Bmagnitude = 20*np.log(np.abs(Bshifted)) #only for visualizing purposes 
    Btransformed = np.asarray(Bmagnitude, np.uint8)
    
    Gfourier = np.fft.fft2(G)
    Gshifted = np.fft.fftshift(Gfourier)
    Gmagnitude = 20*np.log(np.abs(Gshifted)) 
    Gtransformed = np.asarray(Gmagnitude, np.uint8)
    
    Rfourier = np.fft.fft2(R)
    Rshifted = np.fft.fftshift(Rfourier)
    Rmagnitude = 20*np.log(np.abs(Rshifted))
    Rtransformed = np.asarray(Rmagnitude, np.uint8)

    Bfiltered = passFilter(Bshifted,"Lowpass",100)
    Bfiltered = boxFilter(Bfiltered,(number_of_rows//2,number_of_columns//2),(65,65 ))
    Bfiltered  = Bfiltered - Bshifted
    Bfiltered = bandRejectFilter(Bfiltered, (100,430))
    
    Gfiltered = passFilter(Gshifted,"Lowpass", 100)
    #Gfiltered = passFilter(Gshifted,"Highpass", 50)
    Gfiltered = boxFilter(Gfiltered,(number_of_rows//2,number_of_columns//2),(105,105 ))
    
    Gfiltered = boxFilter(Gfiltered,(number_of_rows//2,number_of_columns//4),(4,number_of_rows//2-2 ))  # left horizontal line
    Gfiltered = boxFilter(Gfiltered,(number_of_rows//4,number_of_columns//2),(number_of_columns//2-1  ,4))  # upper vertical line
    Gfiltered = boxFilter(Gfiltered,(int(number_of_rows*3/4),number_of_columns//2),(number_of_columns//2  ,4))  # bottom vertical line
    Gfiltered = boxFilter(Gfiltered,(number_of_rows//2,int(number_of_columns*3/4)),(4,number_of_rows//2 ))  # right horizontal line
    Gfiltered  = Gfiltered - Gshifted
    Gfiltered = bandRejectFilter(Gfiltered, (100,430))
    
    Rfiltered = passFilter(Rshifted,"Lowpass", 100)
    #Rfiltered = passFilter(Rshifted,"Highpass", 20)
    Rfiltered = boxFilter(Rfiltered,(number_of_rows//2,number_of_columns//2),(85,85 ))
    
    Rfiltered = boxFilter(Rfiltered,(number_of_rows//2,number_of_columns//4),(4,number_of_rows//2 -2))  # left horizontal line
    Rfiltered = boxFilter(Rfiltered,(number_of_rows//4,number_of_columns//2),(number_of_columns//2 -2 ,4))  # upper vertical line
    Rfiltered = boxFilter(Rfiltered,(int(number_of_rows*3/4),number_of_columns//2),(number_of_columns//2 -2 ,4))  # bottom vertical line
    Rfiltered = boxFilter(Rfiltered,(number_of_rows//2,int(number_of_columns*3/4)),(4,number_of_rows//2 -2))  # right horizontal line
    Rfiltered  = Rfiltered - Rshifted
    Rfiltered = bandRejectFilter(Rfiltered, (100,430))
    #Rfiltered = bandRejectFilter(Rfiltered, (65,100))
    """
    Bmagnitude = 20*np.log(np.abs(Bfiltered)) #only for visualizing purposes 
    Btransformed = np.asarray(Bmagnitude, np.uint8)
    Gmagnitude = 20*np.log(np.abs(Gfiltered)) #only for visualizing purposes 
    Gtransformed = np.asarray(Gmagnitude, np.uint8)
    Rmagnitude = 20*np.log(np.abs(Rfiltered)) #only for visualizing purposes 
    Rtransformed = np.asarray(Rmagnitude, np.uint8)
    filtered = np.concatenate((Btransformed,Gtransformed,Rtransformed),1)
    cv2.imshow("All Filtered", filtered)
    cv2.imwrite(output_path+"/4Filtered.png", filtered)
    """
    # Apply inverse shift
    Breshifted = np.fft.ifftshift(Bfiltered)
    Greshifted = np.fft.ifftshift(Gfiltered)
    Rreshifted = np.fft.ifftshift(Rfiltered)
    # Return to spatial domain using Inverse Fourier Transformation
    Binversefourier = np.fft.ifft2(Breshifted)
    Binversefourier = np.abs(Binversefourier)
    Ginversefourier = np.fft.ifft2(Greshifted)
    Ginversefourier = np.abs(Ginversefourier)
    Rinversefourier = np.fft.ifft2(Rreshifted)
    Rinversefourier = np.abs(Rinversefourier)
    # Normalize
    Binversefourier -= Binversefourier.min()
    Binversefourier = Binversefourier*255 / Binversefourier.max()
    Ginversefourier -= Ginversefourier.min()
    Ginversefourier = Ginversefourier*255 / Ginversefourier.max()
    Rinversefourier -= Rinversefourier.min()
    Rinversefourier = Rinversefourier*255 / Rinversefourier.max()
    
    Benhanced = Binversefourier.astype(np.uint8)
    Genhanced = Ginversefourier.astype(np.uint8)
    Renhanced = Rinversefourier.astype(np.uint8)
    enhanced = cv2.merge((Benhanced,Genhanced,Renhanced))
    #result = np.concatenate((Benhanced,Genhanced,Renhanced),1)
    #cv2.imshow("All Back to Spatial Domain", result)
    cv2.imwrite(output_path+"/Enhanced.png", enhanced)
    return enhanced

def BGR2YCrCb(image):
    #JPEG conversion (Y, CB and CR have the full 8-bit range of [0...255])  https://en.wikipedia.org/wiki/YCbCr
    #image is BGR
    converted = np.zeros(image.shape, np.uint8) 
    coefficients = [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]]
    converted[:,:,0] =   coefficients[0][0]*image[:,:,2] + coefficients[0][1]*image[:,:,1] + coefficients[0][2]*image[:,:,0]  
    converted[:,:,1] =   coefficients[1][0]*image[:,:,2] + coefficients[1][1]*image[:,:,1] + coefficients[1][2]*image[:,:,0] +128
    converted[:,:,2] =   coefficients[2][0]*image[:,:,2] + coefficients[2][1]*image[:,:,1] + coefficients[2][2]*image[:,:,0] +128
    
    return converted
def YCrCb2BGR(image):
    #JPEG conversion (Y, CB and CR have the full 8-bit range of [0...255])  https://en.wikipedia.org/wiki/YCbCr
    #image is YCrCb
    converted = np.zeros(image.shape, np.uint8) 
    #B
    converted[:,:,0] =  image[:,:,0] + 1.402 * (image[:,:,1]-128)
    #G
    converted[:,:,1] =  image[:,:,0] - 0.344136 * (image[:,:,2]-128) - 0.714136 * (image[:,:,1]-128)
    #R
    converted[:,:,2] =  image[:,:,0] + 1.772 * (image[:,:,2]-128)
    
    return converted

def convolution2D(image, filter):
    m, n = filter.shape
    if (m == n): # check if the given filter(kernel) is square or not
        size = image.shape
    
        output = np.zeros(size, np.uint8)           
        # Loop over entries (y,x)
        for i in range((m-1)//2, size[0] - ((m-1)//2) ):  # Due to the nature of the convolution operation, endmost m-1 rows and m-1 columns will not be calculated
            for j in range((m-1)//2, size[1] - (m-1)//2 ):  # use // to get an integer result not float (4/2 = 2.0 where as 4//2 = 2)
                output[i][j] =(1)*round(np.sum(image[i-((m-1)//2):i+1-((m-1)//2), j-((m-1)//2):j+1-((m-1)//2)]*filter))
                                    
    return output

def DCT(block):
    blocksize = block.shape[0]
    dct = np.zeros((blocksize,blocksize),np.float32 ) 
    for u in range(0,blocksize):
        for v in range(0,blocksize):
            summation = 0
            for i in range(0,blocksize):
                for j in range(0,blocksize):
                    summation += block[i][j]*np.cos(np.pi*u*(2*i+1)/(2*blocksize))*np.cos(np.pi*v*(2*j+1)/(2*blocksize))
            if(u == 0):
                c1 = 1/np.sqrt(2)
            else:
                c1 = 1
            if(v == 0):
                c2 = 1/np.sqrt(2)
            else:
                c2 = 1
            summation *= c1*c2

            dct[u][v] = summation*np.sqrt(1/(2*blocksize))

    return dct

def IDCT(block):
    blocksize = block.shape[0]
    idct = np.zeros((blocksize,blocksize),np.float32 ) 
    for i in range(0,blocksize):
        for j in range(0,blocksize):
            summation = 0
            for u in range(0,blocksize):
                for v in range(0,blocksize):
                    summation += block[u][v]*np.cos(np.pi*u*(2*i+1)/(2*blocksize))*np.cos(np.pi*v*(2*j+1)/(2*blocksize))
                    if(u == 0):
                        c1 = 1/np.sqrt(2)
                    else:
                        c1 = 1
                    if(v == 0):
                        c2 = 1/np.sqrt(2)
                    else:
                        c2 = 1
                    summation *= c1*c2
            idct[i][j] = summation*np.sqrt(1/(2*blocksize)) + np.sqrt(1/(4*blocksize))*block[0][0]
    #print("dct: ",dct)
    return idct

def zigZag(block):
    blocksize = block.shape[0]
    output = [[] for x in range(blocksize*2 -1)]
    # loop over matrix entries
    for i in range(blocksize):
        for j in range(blocksize):
            temp = i+j
            if(temp%2 ==0):
                #add at the beginning of the list
                output.insert(0,block[i][j])
            else:
                #add at the end of the list
                (output[temp]).append(block[i][j])
                
    return output

def runLengtEncoding(alist):
    i = 0
    n = len(alist)
    newlist = []
    while i < n:
        count = 1
        while (i < n-1 and alist[i] == alist[i+1]):
            count += 1
            i += 1
        i += 1
        newlist.append((alist[i-1],count))
        
    return newlist

def complement(bits):  #1'st complement bit representation for negative integers
    complement = ""
    for x in range(0,len(bits)):
        if bits[x] == '0':
            complement += "1"
        else:
            complement += "0"

    return complement

def huffmanEncoding(array):
    
    lookUpTableAC = [[(0,0),"1010"],[(0,1), "00"],[(0,2),"01"],[(0,3),"100"],[(0,4),"1011"],[(0,5),"11010"],[(0,6),"1111000"],[(0,7),"11111000"],[(0,8),"1111110110"],[(0,9),"1111111110000010"],[(0,10),"1111111110000011"],
                     [(1,1),"1100"],[(1,2),"11011"],[(1,3),"1111001"],[(1,4),"111110110"],[(1,5),"11111110110"],[(1,6),"1111111110000100"],[(1,7),"1111111110000101"],[(1,8),"1111111110000110"],[(1,9),"1111111110000111"],[(1,10),"1111111110001000"],
                     [(2,1),"11100"],[(2,2),"11111001"],[(2,3),"1111110111"],[(2,4),"111111110100"],[(2,5),"1111111110001001"],[(2,6),"1111111110001010"],[(2,7),"1111111110001011"],[(2,8),"1111111110001100"],[(2,9),"1111111110001101"],[(2,10),"1111111110001110"],
                     [(3,1),"111010"],[(3,2),"111110111"],[(3,3),"111111110101"],[(3,4),"1111111110001111"],[(3,5),"1111111110010000"],[(3,6),"1111111110010001"],[(3,7),"1111111110010010"],[(3,8),"1111111110010011"],[(3,9),"1111111110010100"],[(3,10),"1111111110010101"],
                     [(4,1),"111011"],[(4,2),"1111111000"],[(4,3),"1111111110010110"],[(4,4),"1111111110010111"],[(4,5),"1111111110011000"],[(4,6),"1111111110011001"],[(4,7),"1111111110011010"],[(4,8),"1111111110011011"],[(4,9),"1111111110011100"],[(4,10),"1111111110011101"],
                     [(5,1),"1111010"],[(5,2),"11111110111"],[(5,3),"1111111110011110"],[(5,4),"1111111110011111"],[(5,5),"1111111110100000"],[(5,6),"1111111110100001"],[(5,7),"1111111110100010"],[(5,8),"1111111110100011"],[(5,9),"1111111110100100"],[(5,10),"1111111110100101"],
                     [(6,1),"1111011"],[(6,2),"111111110110"],[(6,3),"1111111110100110"],[(6,4),"1111111110100111"],[(6,5),"1111111110101000"],[(6,6),"1111111110101001"],[(6,7),"1111111110101010"],[(6,8),"1111111110101011"],[(6,9),"1111111110101100"],[(6,10),"1111111110101101"],
                     [(7,1),"11111010"],[(7,2),"111111110111"],[(7,3),"1111111110101110"],[(7,4),"1111111110101111"],[(7,5),"1111111110110000"],[(7,6),"1111111110110001"],[(7,7),"1111111110110010"],[(7,8),"1111111110110011"],[(7,9),"1111111110110100"],[(7,10),"1111111110110101"],
                     [(8,1),"111111000"],[(8,2),"111111111000000"],[(8,3),"1111111110110110"],[(8,4),"1111111110110111"],[(8,5),"1111111110111000"],[(8,6),"1111111110111001"],[(8,7),"1111111110111010"],[(8,8),"1111111110111011"],[(8,9),"1111111110111100"],[(8,10),"1111111110111101"],
                     [(9,1),"111111001"],[(9,2),"1111111110111110"],[(9,3),"1111111110111111"],[(9,4),"1111111111000000"],[(9,5),"1111111111000001"],[(9,6),"1111111111000010"],[(9,7),"1111111111000011"],[(9,8),"1111111111000100"],[(9,9),"1111111111000101"],[(9,10),"1111111111000110"],
                     [(10,1),"111111010"]]
    
    lookUpTableDC = [[(0,0),"00"],[(0,1),"010"],[(0,2),"011"],[(0,3),"100"],[(0,4),"101"],[(0,5),"110"],[(0,6),"1110"],[(0,7),"11110"],[(0,8),"111110"],[(0,9),"1111110"],[(0,10),"11111110"],[(0,11),"111111110"]]
    output = []
    zerocount = 0
    for x in range(0,array.shape[0]):
        
        if (array[x][0] == 0):
            zerocount += array[x][1] 
        else:
            #print("binary representation of {} is {} with length {}".format(array[x][0],bin(array[x][0])[2:],len(bin(array[x][0])[2:])))
            if array[x][0] < 0 :
                output.append(((zerocount, len(bin(array[x][0])[3:])),array[x][0]))
            else:
                output.append(((zerocount, len(bin(array[x][0])[2:])),array[x][0]))
                
            zerocount = 0
        if (x == array.shape[0]-1) : # last element in the list
            output.append((0,0))
    #print("output: ",output)

    huffman = []
    bitstream = ""
    for coeff in range (0,len(output)):
        if (coeff == len(output)-1): # end of the 8x8 block's is (0,0) coded as "1010"
            huffman.append(("1010"))
            bitstream += "1010"
        elif (coeff == 0):
            for codeword in lookUpTableDC:
                if codeword[0] == output[coeff][0]:
                         
                    if output[coeff][1] < 0:
                        huffman.append((codeword[1],complement(bin(output[coeff][1])[3:])))
                        bitstream += codeword[1] + complement(bin(output[coeff][1])[3:])
                    else:
                        huffman.append((codeword[1],bin(output[coeff][1])[2:]))
                        bitstream += codeword[1] + bin(output[coeff][1])[2:]
        
        else:
            for codeword in lookUpTableAC:
                if codeword[0] == output[coeff][0]:
                         
                    if output[coeff][1] < 0:
                        huffman.append((codeword[1],complement(bin(output[coeff][1])[3:])))
                        bitstream += codeword[1] + complement(bin(output[coeff][1])[3:])
                    else:
                        huffman.append((codeword[1],bin(output[coeff][1])[2:]))
                        bitstream += codeword[1] + bin(output[coeff][1])[2:]
    return bitstream
          
def the2_write(input_img_path, output_path):
    # Read the input image
    img = cv2.imread(input_img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (480,480), interpolation= cv2.INTER_LINEAR)
    number_of_rows = img.shape[0]
    number_of_columns = img.shape[1]

    create_output_path(output_path)
    (B, G, R) = cv2.split(img)
    # Convert the color space
    YCrCb = BGR2YCrCb(img)
    (Y, Cr, Cb) = cv2.split(YCrCb)
    # Subsample the chrominance channels Cr and Cb
    kernel = np.array([[1/4, 1/4], [1/4, 1/4]])
    Cr = convolution2D(Cr, kernel)[::2,::2] # Cb and Cr are each subsampled at a factor of 2 both horizontally and vertically.
    Cb = convolution2D(Cb, kernel)[::2,::2] # Now for every 4 Y pixels, there will exist only 1 CbCr pixel.
    # Quantization matrices
    QY = 1/np.array([[16,11,10,16,24,40,51,61],
                   [12,12,14,19,26,48,60,55],
                   [14,13,16,24,40,57,69,56],
                   [14,17,22,29,51,87,80,62],
                   [18,22,37,56,68,109,103,77],
                   [24,35,55,64,81,104,113,92],
                   [49,64,78,87,103,121,120,101],
                   [72,92,95,98,112,100,103,99]])

    QC = 1/np.array([[17,18,24,47,99,99,99,99],
                     [18,21,26,66,99,99,99,99],
                     [24,26,56,99,99,99,99,99],
                     [47,66,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99]])
    scale = 1
    QY *= scale
    QC *= scale
    # Each of the three Y,Cb and Cr components will be transformed, quantized and encoded separately
    # First divide each Y, Cr, Cv into sub-images each consist of 8x8 pixels. Then we aplly the operations
    blocksize = 8
    with open(output_path+"/compressed.txt", "w") as file:
        file.write("Y {} {} \n".format(number_of_rows,number_of_columns))
        for r in range(0, Y.shape[0]//8):
            for c in range(0, Y.shape[1]//8):
                block = Y[r*blocksize : (r+1)*blocksize, c*blocksize: (c+1)*blocksize]
                #print("block: ",block)
                # Convert datatype to int16 so we can subtract 128 from each pixel
                block = block.astype(np.int16)-128
                #print("8x8 scaled block: ",block)
                block = block.astype(np.float32)
                block = cv2.dct(block)
                quantized = block * QY
                #quantized = np.multiply(DCT(block), QY)
                #print("DCT: ",DCT(block))
                quantized = quantized.astype(np.int16)
                #print("quantized: ",quantized)
                zigzag_temp = [[] for x in range(blocksize*2 -1)]
                # Zigzag traversal
                for i in range(blocksize):
                    for j in range(blocksize):
                        temp = i+j
                        if(temp%2 ==0):
                            #add at the beginning of the list
                            zigzag_temp[temp].insert(0,quantized[i][j])
                        else:
                            #add at the end of the list
                            zigzag_temp[temp].append(quantized[i][j])
                #print("zigzag_temp: ",zigzag_temp)
                
                zigzag = []
                for sublist in zigzag_temp:
                    #print("i: ",sublist)
                    for element in range(0,len(sublist)):
                        #print("j: ",sublist[element])
                        zigzag.append(sublist[element])
                #print("zigzag: ",zigzag)
                # Runlength encoding 
                zigzag = np.asarray(runLengtEncoding(zigzag))
                #print("runLengtEncoding: ",zigzag)
                # Apply Huffman Coding to 8x8 blocks and write the output bitstream representing 8x8 block to a txt file.
                file.write(huffmanEncoding(zigzag)+"\n")
                #print(c)
                #print("Huffman: ",huffmanEncoding(zigzag))
        file.write("Cr\n".format(number_of_rows//4,number_of_columns//4))
        for r in range(0, Cr.shape[0]//8):
            for c in range(0, Cr.shape[1]//8):
                block = Cr[r*blocksize : (r+1)*blocksize, c*blocksize: (c+1)*blocksize]
                # Convert datatype to int16 so we can subtract 128 from each pixel
                block = block.astype(np.int16)-128
                block = block.astype(np.float32)
                block = cv2.dct(block)
                quantized = block * QC
                #print("8x8 scaled block: ",block)
                #quantized = np.multiply(DCT(block), QC)
                quantized = quantized.astype(np.int16)
                #print("Transformed and quantized: ",quantized)
                zigzag_temp = [[] for x in range(blocksize*2 -1)]
                # Zigzag traversal
                for i in range(blocksize):
                    for j in range(blocksize):
                        temp = i+j
                        if(temp%2 ==0):
                            #add at the beginning of the list
                            zigzag_temp[temp].insert(0,quantized[i][j])
                        else:
                            #add at the end of the list
                            zigzag_temp[temp].append(quantized[i][j])
                #print("zigzag_temp: ",zigzag_temp)
                
                zigzag = []
                for sublist in zigzag_temp:
                    #print("i: ",sublist)
                    for element in range(0,len(sublist)):
                        #print("j: ",sublist[element])
                        zigzag.append(sublist[element])
                #print("zigzag: ",zigzag)
                # Runlength encoding 
                zigzag = np.asarray(runLengtEncoding(zigzag))
                #print("runLengtEncoding: ",zigzag)
                # Apply Huffman Coding to 8x8 blocks and write the output bitstream representing 8x8 block to a txt file.
                file.write(huffmanEncoding(zigzag)+"\n")
                
        file.write("Cb\n".format(number_of_rows//4,number_of_columns//4))
        for r in range(0, Cb.shape[0]//8):
            for c in range(0, Cb.shape[1]//8):
                block = Cb[r*blocksize : (r+1)*blocksize, c*blocksize: (c+1)*blocksize]
                # Convert datatype to int16 so we can subtract 128 from each pixel
                block = block.astype(np.int16)-128
                block = block.astype(np.float32)
                block = cv2.dct(block)
                quantized = block * QC
                #print("8x8 scaled block: ",block)
                #quantized = np.multiply(DCT(block), QC)
                quantized = quantized.astype(np.int16)
                #print("Transformed and quantized: ",quantized)
                zigzag_temp = [[] for x in range(blocksize*2 -1)]
                # Zigzag traversal
                for i in range(blocksize):
                    for j in range(blocksize):
                        temp = i+j
                        if(temp%2 ==0):
                            #add at the beginning of the list
                            zigzag_temp[temp].insert(0,quantized[i][j])
                        else:
                            #add at the end of the list
                            zigzag_temp[temp].append(quantized[i][j])
                #print("zigzag_temp: ",zigzag_temp)
                
                zigzag = []
                for sublist in zigzag_temp:
                    #print("i: ",sublist)
                    for element in range(0,len(sublist)):
                        #print("j: ",sublist[element])
                        zigzag.append(sublist[element])
                #print("zigzag: ",zigzag)
                # Runlength encoding 
                zigzag = np.asarray(runLengtEncoding(zigzag))
                #print("runLengtEncoding: ",zigzag)
                # Apply Huffman Coding to 8x8 blocks and write the output bitstream representing 8x8 block to a txt file.
                file.write(huffmanEncoding(zigzag)+"\n")
                
    file.close()
    img_size = os.stat(input_img_path).st_size
    compressed_size = os.stat(output_path+"/compressed.txt").st_size
    compression_ratio = img_size/compressed_size
    #print("Original size: {}, compressed size: {}, compression ratio: {}".format(img_size, compressed_size, compression_ratio))
    print("{}, {}, {}".format(img_size, compressed_size, compression_ratio))
    return output_path+"/compressed.txt"
    
def the2_read(input_img_path):
    lookUpTableAC = [[(0,0),"1010"],[(0,1), "00"],[(0,2),"01"],[(0,3),"100"],[(0,4),"1011"],[(0,5),"11010"],[(0,6),"1111000"],[(0,7),"11111000"],[(0,8),"1111110110"],[(0,9),"1111111110000010"],[(0,10),"1111111110000011"],
                     [(1,1),"1100"],[(1,2),"11011"],[(1,3),"1111001"],[(1,4),"111110110"],[(1,5),"11111110110"],[(1,6),"1111111110000100"],[(1,7),"1111111110000101"],[(1,8),"1111111110000110"],[(1,9),"1111111110000111"],[(1,10),"1111111110001000"],
                     [(2,1),"11100"],[(2,2),"11111001"],[(2,3),"1111110111"],[(2,4),"111111110100"],[(2,5),"1111111110001001"],[(2,6),"1111111110001010"],[(2,7),"1111111110001011"],[(2,8),"1111111110001100"],[(2,9),"1111111110001101"],[(2,10),"1111111110001110"],
                     [(3,1),"111010"],[(3,2),"111110111"],[(3,3),"111111110101"],[(3,4),"1111111110001111"],[(3,5),"1111111110010000"],[(3,6),"1111111110010001"],[(3,7),"1111111110010010"],[(3,8),"1111111110010011"],[(3,9),"1111111110010100"],[(3,10),"1111111110010101"],
                     [(4,1),"111011"],[(4,2),"1111111000"],[(4,3),"1111111110010110"],[(4,4),"1111111110010111"],[(4,5),"1111111110011000"],[(4,6),"1111111110011001"],[(4,7),"1111111110011010"],[(4,8),"1111111110011011"],[(4,9),"1111111110011100"],[(4,10),"1111111110011101"],
                     [(5,1),"1111010"],[(5,2),"11111110111"],[(5,3),"1111111110011110"],[(5,4),"1111111110011111"],[(5,5),"1111111110100000"],[(5,6),"1111111110100001"],[(5,7),"1111111110100010"],[(5,8),"1111111110100011"],[(5,9),"1111111110100100"],[(5,10),"1111111110100101"],
                     [(6,1),"1111011"],[(6,2),"111111110110"],[(6,3),"1111111110100110"],[(6,4),"1111111110100111"],[(6,5),"1111111110101000"],[(6,6),"1111111110101001"],[(6,7),"1111111110101010"],[(6,8),"1111111110101011"],[(6,9),"1111111110101100"],[(6,10),"1111111110101101"],
                     [(7,1),"11111010"],[(7,2),"111111110111"],[(7,3),"1111111110101110"],[(7,4),"1111111110101111"],[(7,5),"1111111110110000"],[(7,6),"1111111110110001"],[(7,7),"1111111110110010"],[(7,8),"1111111110110011"],[(7,9),"1111111110110100"],[(7,10),"1111111110110101"],
                     [(8,1),"111111000"],[(8,2),"111111111000000"],[(8,3),"1111111110110110"],[(8,4),"1111111110110111"],[(8,5),"1111111110111000"],[(8,6),"1111111110111001"],[(8,7),"1111111110111010"],[(8,8),"1111111110111011"],[(8,9),"1111111110111100"],[(8,10),"1111111110111101"],
                     [(9,1),"111111001"],[(9,2),"1111111110111110"],[(9,3),"1111111110111111"],[(9,4),"1111111111000000"],[(9,5),"1111111111000001"],[(9,6),"1111111111000010"],[(9,7),"1111111111000011"],[(9,8),"1111111111000100"],[(9,9),"1111111111000101"],[(9,10),"1111111111000110"],
                     [(10,1),"111111010"]]
    lookUpTableDC = [[(0,0),"00"],[(0,1),"010"],[(0,2),"011"],[(0,3),"100"],[(0,4),"101"],[(0,5),"110"],[(0,6),"1110"],[(0,7),"11110"],[(0,8),"111110"],[(0,9),"1111110"],[(0,10),"11111110"],[(0,11),"111111110"]]

    QY = np.array([[16,11,10,16,24,40,51,61],
                   [12,12,14,19,26,48,60,55],
                   [14,13,16,24,40,57,69,56],
                   [14,17,22,29,51,87,80,62],
                   [18,22,37,56,68,109,103,77],
                   [24,35,55,64,81,104,113,92],
                   [49,64,78,87,103,121,120,101],
                   [72,92,95,98,112,100,103,99]])

    QC = np.array([[17,18,24,47,99,99,99,99],
                     [18,21,26,66,99,99,99,99],
                     [24,26,56,99,99,99,99,99],
                     [47,66,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99],
                     [99,99,99,99,99,99,99,99]])
    scale = 1
    QY *= scale
    QC *= scale
    
    image = np.zeros((1536,2048,3),dtype=np.uint8)
    channel = 0
    blocksize = 8
    size = []
    with open(input_img_path, "r") as file:   
        lines = file.readlines()
        for line in lines:
            if("Y" in line):
                #print("channel0")
                channel = 0
                Q = QY
                r,c = 0,0
                size = line.split(" ")[1:3]
                #print("size: ",size)
                cmax = int(size[1])//blocksize
                #print("cmax: ",cmax)
            elif("Cr" in line):
                #print("channel1")
                channel = 1
                Q = QC
                r,c = 0,0
                cmax = cmax//2
                #print("cmax: ",cmax)
            elif("Cb" in line):
                #print("channel2")
                channel = 2
                Q = QC 
                r,c = 0,0
            else:
                #print("line: ",line)
                decode = []
                cursor = 0
                count = 0
                for x in range(0,len(line)):
                    #print("line[{}:{}]: {} ".format(cursor,x,line[cursor:x]))
                    if count == 0 :
                        lookUpTable = lookUpTableDC
                        #print("lookupDC")
                    else:
                        #print("lookupAC")
                        lookUpTable = lookUpTableAC
                    for code in lookUpTable:
                        if(line[cursor:x] == code[1]):
                            count= 1
                            if(code[1]== "1010"):#end of the 8x8 block
                                decode.append((0,0))
                                #print("8x8 done: ", decode)
                            else:
                                #print("code: ",code[1])
                                symbol1 = code[0]
                                #print("symbol1: ",symbol1)
                                    
                                value = line[x:x+symbol1[1]]
                                #print("value: ", value)
                                if (len(value)== 0 or value == "\n"):
                                    break
                                if value[0] == '0':
                                    #print("negative")
                                    value = int(complement(value),2)*(-1)
                                else:
                                    value = int(value,2)
                                #print("value: ",value)
                            decode.append((symbol1,value))
                            cursor = x+symbol1[1]
                    
                # reconstruct zigzag 
                length = 0
                zigzag = []
                for symbols in decode:
                    if(symbols == (0,0)):
                        for i in range(0,64-length):
                            zigzag.append(0)
                        break
                    else:
                        for i in range(0,(symbols[0])[0]):
                            zigzag.append(0)

                        zigzag.append(symbols[1])
                        length = len(zigzag)
                #print("zigzag: ",zigzag)
                
                quantized = np.zeros((blocksize,blocksize),np.int16)
                if (len(zigzag) == 64):
                    tempx,tempy = 0,0
                    for i in range(0,63):
                        quantized[tempx][tempy] = zigzag[i]
                        #print("tempx,tempy: ",tempx,tempy)
                        if((tempx+tempy)%2 == 0): #traversal will start from bottom to top
                            if tempx != 0:
                                dx = -1
                            else:
                                dx = 0
                            if tempy != 7:
                                dy = 1
                            else:
                                dy = 0 
                        else: #traversal will start from top to bottom
                            if tempx != 7:
                                dx = 1
                            else:
                                dx = 0
                            if tempy != 0:
                                dy = -1
                            else:
                                dy = 0
                        #print("dx,dy: ",dx,dy)
                        tempx += dx
                        tempy += dy
                #print("quantized:\n ",quantized)
                dequantized = quantized * Q
                #print("dequantized: ",dequantized)
                dequantized = dequantized.astype(np.float32)
                block = cv2.idct(dequantized)
                #block = IDCT(dequantized)
                block = block.astype(np.int16)
                #print("block: ",block)
                block = block +128
                #print("rescaled: ",block)
                #print("channel: ",channel)
                #print("reconstructing [",channel,"] [", r*blocksize ,":",(r+1)*blocksize,"] [", c*blocksize,":",(c+1)*blocksize,"]")
                image[r*blocksize : (r+1)*blocksize,c*blocksize: (c+1)*blocksize,channel] = block
                if (c==cmax-1):
                    c = 0
                    r+= 1
                else:
                    c+= 1
                           
    file.close()    
    #show image
    Y = image[:,:,0]
    Cr = image[:,:,1]
    Cb = image[:,:,2]
    Cr4 = Cr[0:int(size[0])//2,0:int(size[1])//2]
    Cr = cv2.resize(Cr4, (2048, 1536), interpolation= cv2.INTER_LINEAR)
    Cb4 = Cb[0:int(size[0])//2,0:int(size[1])//2]
    Cb = cv2.resize(Cb4, (2048, 1536), interpolation= cv2.INTER_LINEAR)
    #cv2.imshow("Y",Y)
    #cv2.imshow("Cr",Cr)
    #cv2.imshow("Cb",Cb)
    image[:,:,2] = Cr
    image[:,:,1] = Cb
    #image = YCrCb2BGR(image)
    image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    cv2.imshow("reconstructed",image)
    #cv2.imwrite("reconstructed.png",image)
"""
np.set_printoptions(suppress=True)  #If True, always print floating point numbers using fixed point notation. If False, then scientific notation is used.

directory = os.getcwd()
print("Writing outputs to: ",directory)
print("please wait until you see \"finished\" text")
part1('./THE2_Images/1.png','./THE2outputs/Part1/1/')
part1('./THE2_Images/2.png','./THE2outputs/Part1/2/')
enhance_3('./THE2_images/3.png','./THE2outputs/Part2/3/')
enhance_4('./THE2_images/4.png','./THE2outputs/Part2/4/')
the2_write('./THE2_images/5.png','./THE2_images/compression_outputs')
the2_read('./THE2_images/compression_outputs/compressed.txt')
print("finished")
"""


