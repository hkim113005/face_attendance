import os
import cv2

path = 'dataset'

imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

for imagePath in imagePaths:
    if ".DS_Store" in imagePath:
        continue
    
    cv2.namedWindow(imagePath, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(imagePath, 400, 400)
   
    try:
        img = cv2.imread(imagePath)
    except:
        continue

    cv2.imshow(imagePath, img)

    k = 0
    while True:
        k = cv2.waitKey(0)
        if k == 27:
            break
        elif k == 115:
            break
        elif k == 100:
            os.remove(imagePath)
            break
    if k == 27:
        break

    cv2.destroyAllWindows()