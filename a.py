import cv2
import numpy as np

address = "https://192.168.1.2:8080/video"
cap = cv2.VideoCapture(0)
cap.open(address)
# Seta a resolução para 4K
cap.set(3, 1920)
cap.set(4, 1080)

#874 418 373 159

_, img = cap.read()
    
img = img[418:418+159, 874:874+373]
img2 = img.copy()

kernel = tuple((3, 3))
kernelF = tuple((3, 1))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
filtro = cv2.filter2D(img2, -1, kernelF)
gradient = cv2.morphologyEx(filtro, cv2.MORPH_GRADIENT, kernel)
opening = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(closed, cv2.MORPH_GRADIENT, kernel)

canny = cv2.Canny(closed, 75,
                    255)


""" cv2.imshow("closed", closed)
cv2.imshow("gradient", gradient)
cv2.imshow("filtro", filtro)
cv2.imshow("canny", canny) """

original, hierarchy = cv2.findContours(
    closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

compare_img = np.zeros(img.shape)
for i in range(len(original)):
    # Internal = !=
    # External = ==
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(compare_img, original, i, 255, 1)
        
cv2.imshow("Imagem para comparação", compare_img)
cv2.waitKey(0)
    

while True:
    _, img = cap.read()
    
    img = img[418:418+159, 874:874+373]
    img2 = img.copy()
    
    kernel = tuple((3, 3))
    kernelF = tuple((3, 1))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    filtro = cv2.filter2D(img2, -1, kernelF)
    gradient = cv2.morphologyEx(filtro, cv2.MORPH_GRADIENT, kernel)
    opening = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(closed, cv2.MORPH_GRADIENT, kernel)

    canny = cv2.Canny(closed, 75,
                        255)


    cv2.imshow("closed", closed)
    cv2.imshow("gradient", gradient)
    cv2.imshow("filtro", filtro)
    cv2.imshow("canny", canny)

    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    cnt1 = np.asarray(original)
    cnt2 = np.asarray(contours)
    final = np.zeros(img.shape)
    print(len(cnt1), len(cnt2))
    matches = cv2.matchShapes(cnt1[0], cnt2[0], 1, 0.0)
    print(matches, len(contours), len(original))
    if len(contours) < len(original):
        contour_range = len(contours)
    else:
        contour_range = len(original)
        
    total = 0
    
    for i in range(contour_range):
        matches = cv2.matchShapes(cnt1[i], cnt2[i], 1, 0.0)
        total += matches

        
    for i in range(len(contours)):
        # Internal = !=
        # External = ==
        if total / contour_range < .15:
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(img, contours, i, (0,255,0), 1)
        else:
            if hierarchy[0][i][3] == -1:
                cv2.drawContours(img, contours, i, (0,0,255), 1)

    cv2.imshow("img", img)
    cv2.waitKey(1)
