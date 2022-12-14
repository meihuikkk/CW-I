import sys
import cv2 as cv
import numpy as np



# img = cv.imread('NoEntry5.bmp',0)
# img = cv.medianBlur(img,5)
# cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
# circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=1,maxRadius=50)
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
# cv.imshow('detected circles',cimg)
# cv.waitKey(0)
# cv.destroyAllWindows()



def main(argv):
    default_file = 'NoEntry13.bmp'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows/10,
                              param1=20, param2=50,
                              minRadius=5, maxRadius=80)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)

    cv.imshow("detected circles", src)
    cv.imwrite("no_entry.jpg",src)
    cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])