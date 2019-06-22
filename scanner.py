import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt


def image_read(image_address):
    image = cv2.imread(image_address)
    # Scale down/up image to a size that opencv can work with
    image = cv2.resize(image, (1200, 800))
    return image


def image_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def image_blur(image):
    blurred_image = cv2.GaussianBlur(image, KERNEL_SIZE, BLUR_INTENSITY)
    return blurred_image


def image_canny(image):
    edged_image = cv2.Canny(image, MIN_THRESH, MAX_THRESH)
    return edged_image


def find_image_contours(image):
    image_contours, image_hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return image_contours, image_hierarchy


def sort_contours(contours):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return sorted_contours


def map_corners(corner):
    corner = corner.reshape((4, 2))
    corner_resize = np.zeros((4, 2), dtype=np.float32)

    add = corner.sum(1)
    corner_resize[0] = corner[np.argmin(add)]
    corner_resize[2] = corner[np.argmax(add)]

    diff = np.diff(corner, axis=1)
    corner_resize[1] = corner[np.argmin(diff)]
    corner_resize[3] = corner[np.argmax(diff)]
    return corner_resize


def find_page_contours(contours):
    for c in contours:
        p = cv2.arcLength(c, True)
        points = cv2.approxPolyDP(c, 0.02*p, True)

        if len(points) == 4:
            target = points
            break
    points = map_corners(target)
    return points


def bird_eye_view(original_image, endpoints):
    pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    perspective = cv2.getPerspectiveTransform(endpoints, pts)
    bird_eye_image = cv2.warpPerspective(original_image, perspective, (800, 800))
    return bird_eye_image


def sharpen_image(bird_eye_image):
    sharpened_image = cv2.filter2D(bird_eye_image, -1, SHARPENING_KERNEL)
    return sharpened_image


KERNEL_SIZE = (5, 5)
BLUR_INTENSITY = 0
MIN_THRESH = 30
MAX_THRESH = 50
SHARPENING_KERNEL = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])

address = sys.argv[1]
original = image_read(address)
gray_scale = image_gray(original)
blurred = image_blur(gray_scale)
edged = image_canny(blurred)
con, h = find_image_contours(edged)
con = sort_contours(con)
approx = find_page_contours(con)
hey = bird_eye_view(original, approx)
hoy = sharpen_image(hey)
cv2.imshow('Image Sharpening', hoy)
cv2.waitKey(0)
