import argparse
import os

import cv2
import math
import numpy as np

def process(input, output):
    ## Read
    img = cv2.imread(input)
    if not img is None:
        ## convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        ## mask of green (36,25,25) ~ (86, 255,255)
        # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
        mask = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))

        ## slice the bandids
        res = cv2.bitwise_and(img, img, mask=mask)

        rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=50)
        thresh = cv2.dilate(thresh, None, iterations=50)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # biggest contour
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(res, [cnt], -1, (0, 255, 0), 3)

        x = 0
        y = 0
        dist = max(thresh.shape[0], thresh.shape[1])
        centerx = thresh.shape[1] / 2
        centery = thresh.shape[0] / 2
        for point in cnt:
            dist_p = math.hypot(centery - point[0][1], centerx - point[0][0])
            if dist_p < dist:
                dist = dist_p
                x = point[0][0]
                y = point[0][1]

        dist_x = abs(x - centerx)
        dist_y = abs(y - centery)
        if dist_x > dist_y:
            if x > centerx:
                crop = img[:, 0:x]
            else:
                crop = img[:, x:]
        else:
            if y > centery:
                crop = img[0:y, :]
            else:
                crop = img[y:, :]

        ## save
        cv2.imwrite(output, crop)


parser = argparse.ArgumentParser(description='Fix tilted images')
parser.add_argument('-i', '--input', type=str, help='Input image path', required=True)
parser.add_argument('-o', '--output', type=str, help='Output image path')

args = vars(parser.parse_args())
print(args)
INPUT = args["input"]
OUTPUT = args["output"]
if OUTPUT is None:
    OUTPUT = INPUT

if os.path.isdir(INPUT):
     # iterate through the names of contents of the folder
    for image_file in os.listdir(INPUT):
        # create the full input path and read the file
        input_path = os.path.join(INPUT, image_file)
        output_file = image_file.split(".")[0] + "_cropped.jpg"
        output_path = os.path.join(OUTPUT, output_file)
        process(input_path, output_path)

else:
    process(INPUT, INPUT.split(".")[0] + "_cropped.jpg")