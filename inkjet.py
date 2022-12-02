import cv2
import numpy as np
import easyocr  
import os
from os import listdir


def rectangle(img):
        #img = cv2.imread(r'C:\Users\hakan.turkmen\Desktop\inkjet_new\inkjet_30888.jpg', 1)
        cv2.rectangle(img,(800,900),(2008,1200),(0,255,0),3)
        
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        
        return img

def crop_image(img, y, height, x, width):
    cropped_part = img[y:y+height, x:x+width]
    # cv2.namedWindow("cropped_part", cv2.WINDOW_NORMAL)
    # cv2.imshow("cropped_part",cropped_part)
    # cv2.waitKey(0)
    
    return cropped_part

def noise_removal(img, kernel_size_1, kernel_size_2):
    kernel = np.ones(kernel_size_1, np.uint8)
    img = cv2.dilate(img,kernel,iterations=1)
    kernel = np.ones(kernel_size_2, np.uint8)
    img = cv2.erode(img,kernel,iterations=1)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (1,1))
    img = cv2.medianBlur(img,3)

    return img

def apply_adaptive_threshold(img, method, blockSize=85, C=3): 
    """
    Apply adaptive thresholding, either Gaussian (threshold value is the weighted sum of neighbourhood values where weights are a Gaussian window) or mean (threshold value is the mean of neighbourhood area). Show result.
    """
    # print("len image shape: ",len(img.shape))
    if len(img.shape) > 2:
        img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)
    else:
        pass
    if method == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif method == 'mean':
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

    img_adaptive = cv2.adaptiveThreshold(
        src=img,
        maxValue=255,
        adaptiveMethod=adaptive_method,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=blockSize,
        C=C,
    )
    return img_adaptive

def adjust_brightness(img, value):
    num_channels = 1 if len(img.shape) < 3 else 1 if img.shape[-1] == 1 else 3
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if num_channels == 1 else img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    print("hsv v değeri:",np.average(v)) #70,75 ver

    if np.average(v) <= 75:
        
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            value = int(-value)
            lim = 0 + value
            v[v < lim] = 0
            v[v >= lim] -= value

    final_hsv = cv2.merge((h, s, v))

    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if num_channels == 1 else img
    return img

def five_read(img):
    trichannel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(trichannel, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, np.array([0, 0, 175]), np.array([179, 255, 255]))
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dlt = cv2.dilate(msk, krn, iterations=1)
    thr = 255 - cv2.bitwise_and(dlt, msk)
    
    return thr

reader = easyocr.Reader(['en'])
path = r"inkjet_images"
test_imgs = [(cv2.imread(os.path.join(path,f)),f) for f in os.listdir(path)]
for img,f in test_imgs:
    # print(f)
    filename = f
    print("---------",filename)

    # rectangle(img)
    img = crop_image(img, y = 900,height = 300, x = 800, width = 1208)
    cv2.namedWindow("crop_image", cv2.WINDOW_NORMAL)
    cv2.imshow("crop_image", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # ret,img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)

    img = adjust_brightness(img, 80)
    # cv2.namedWindow("brig", cv2.WINDOW_NORMAL)
    cv2.imshow("brig", img)
# 
    img = apply_adaptive_threshold(img=img, method='gaussian')
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # img = noise_removal(img, (3,3),(2,2))

   
    img = five_read(img)
    # cv2.namedWindow("five", cv2.WINDOW_NORMAL)
    # cv2.imshow("five", img)


    result = reader.readtext(img,paragraph='True', detail=0)
    #print("easyocr'ın döndürdüğü değer:", result)
    print("result:", result[0])
    find_result = result[0].find("100")
    print("indeks",find_result)
    seri_no = result[0][find_result:find_result+10]
    print("Sonuc:", seri_no)

    


    # cv2.namedWindow("noise_removal", cv2.WINDOW_NORMAL)
    cv2.imshow("noise_removal", img)
    cv2.waitKey(0)


