import cv2
import numpy as np
import os
from math import atan2, cos, sin, sqrt, pi
from cv2 import BORDER_TRANSPARENT
from PIL import Image


def bbox_image(image_file):
    with open(image_file, 'rb') as f:
        original = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        minBGR = (250, 250, 250)
        maxBGR = (255, 255, 255)
        # imageLAB = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        maskBGR = cv2.inRange(original, minBGR, maxBGR)
        sum_vector = maskBGR.sum(axis=0)

        kernel = np.ones((60, 60), np.uint8)
        # dilate maskBGR to take over edges of pixels of color
        maskBGR = cv2.erode(maskBGR, kernel, iterations=2)
        # resultLAB = cv2.bitwise_and(original, original, mask=maskBGR)
        bbox = cv2.boundingRect(maskBGR)
        # print(bbox)
        x, y, w, h = bbox
        if sum(sum_vector[:int(len(sum_vector) / 2)]) > sum(sum_vector[int(len(sum_vector) / 2):]):
            x -= int(0.035 * len(maskBGR[0]))
            w += int(0.035 * len(maskBGR[0]))
        else:
            w += int(0.035 * len(maskBGR[0]))
        return x, y, w, h


def crop_image(x, y, w, h, image_file, output_file):
    with open(image_file, 'rb') as f:
        original = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        crop = original[y:y + h, x:x + w]
        cv2.imwrite(output_file, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 70])


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    # angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    return np.arccos(np.clip(np.dot(np.array([-eigenvectors[0][0], eigenvectors[0][1]]), np.array([0, 1])), -1.0, 1.0)) * np.sign(eigenvectors[0][0])*10, cntr
    # return angle, cntr


def get_rotation_angle(image_file):
    # image = Image.open(image_file)  # open colour image
    # image = image.convert('L').point(lambda band: 255 if band > 250 else 0)  # convert image to black and white
    # image.save(image_file + "_p.png")
    src = cv2.imread(image_file)
    # _, thresh1 = cv2.threshold(src, 250, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour = 0
    max_area = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_area:
            max_area, contour = area, c
        # cv2.drawContours(src, contours, i, (0, 0, 255), 2)
    angle, cntr = getOrientation(contour, src)
    # rotated = imutils.rotate_bound(image, angle)
    print("angle", angle)
    # cv2.drawContours(src, contours, contours.index(contour), (0, 0, 255), 2)
    # rot_mat = cv2.getRotationMatrix2D(cntr, -angle, 1.0)
    rot_mat = cv2.getRotationMatrix2D(cntr, -angle, 1.0)
    return cv2.warpAffine(src, rot_mat, (src.shape[1], src.shape[0]))  # , borderValue=(255, 255, 255))  # , flags=cv2.INTER_LINEAR)
