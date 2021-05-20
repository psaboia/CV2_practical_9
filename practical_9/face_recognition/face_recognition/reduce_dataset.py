# USAGE
# python encode_faces.py --dataset ./datasets/face_recognition_dataset --encodings ../output/encodings.pickle

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import imutils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
#ap.add_argument("-o", "--output", required=True,
#	help="path to output directory of reduced images")
ap.add_argument("-s", "--scale", required=True, type=float, default=0.5,
	help="Scale factor used to resize the image")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
scale = args["scale"]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load
    image = cv2.imread(imagePath)

    # reduce
    reducedImage = cv2.resize(image, (0, 0), fx = scale, fy = scale)

    # save
    root, ext = os.path.splitext(imagePath)
    newImagePath = root + "_r" + ext
    cv2.imwrite(newImagePath, reducedImage)
    print(newImagePath)
