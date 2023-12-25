import cv2 as cv 
import numpy as np 

filename = "house.jpg"
img = cv.imread(filename)
canny = cv.Canny(img, 100, 270)

img1 = img.copy()
img2 = img.copy()

def Hough_transform(image, theta=np.pi/180, treshold=120):
    width, height = image.shape[1], image.shape[0]
    # rho = [0, (width + height) // 2], theta = [0, 2pi]
    num_thetas = round(2*np.pi / theta + 1)
    num_rhos = round((width + height) // 2 + 1)

    # 1. filling the accumulator
    accum = np.zeros((num_rhos, num_thetas))
    for i in range(height):
        for j in range(width):
            if image[i, j] > 20:
                for n in range(num_thetas):
                    x0 = j - width // 2
                    y0 = height // 2 - i
                    r = round(y0 * np.sin(theta * n) + x0 * np.cos(theta * n))
                    accum[r, n] += 1
    # 2. finding the required values
    lines = []
    for r in range(num_rhos):
        for n in range(num_thetas):
            if accum[r, n] > treshold:
                lines.append(np.array([r, theta * n]))
                
    lines = np.array(lines)
    if lines.size == 0: lines = None
    return lines

# My Hough transform
lines = Hough_transform(canny, theta=np.pi/180, treshold=170)
if (lines is not None):
    for i, (rho, theta) in enumerate(lines):
        width, height = canny.shape[1], canny.shape[0]
        a = np.cos(theta)
        b = np.sin(theta)
        
        x0 = a * rho
        y0 = b * rho
        
        x1 = int(x0 + 1000 * (-b)) + width // 2
        y1 = height // 2 - int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b)) + width // 2
        y2 = height // 2 - int(y0 - 1000 * a)
        
        cv.line(img1, (x1, y1), (x2, y2), (0,255,0), thickness=1)
        
    cv.imshow(filename, img1)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Actual Hough transform
lines = cv.HoughLines(canny, rho=1, theta=np.pi/180, threshold=120)
if (lines is not None):
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            
            x0 = a * rho 
            y0 = b * rho 
            
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            
            cv.line(img2, (x1, y1), (x2, y2), (0,0,255), thickness=1)
        
cv.imshow(filename, img2)
cv.waitKey(0)
        