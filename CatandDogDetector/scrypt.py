import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imutils as im
import argparse as arg

def pyramid(image, scale=1.5, minSize = (30, 30)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = im.resize(image, width=w)

        if (image.shape[0] < minSize[1] or image.shape[1] < minSize[0]):
            break
        yield image

def sliding_window(image, stepSize, windowsSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowsSize[1], x:x + windowsSize[0]])


def runGoogleNet(img):
    rows = open(args["labels"]).read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    blob = cv.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
    net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    net.setInput(blob)
    preds = net.forward()
    idxs = np.argsort(preds[0])[::-1][:5]

    for (i, idx) in enumerate(idxs):
        if i == 0:
            text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
            cv.putText(img, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]))

ap = arg.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, 	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

(winW, winH) = (180, 180)

input = cv.imread(args["image"])

cropImg = input[198:922, 70:1440]
cv.imshow("Crop", cropImg)
imgCrop = cropImg.copy()
img1 = imgCrop.copy()

for resized in pyramid(imgCrop, scale=1.5):
    for (x, y, window) in sliding_window(resized, stepSize=180, windowsSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
            print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1, classes[idx], preds[0][idx]))

        rows = open(args["labels"]).read().strip().split("\n")
        classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

        blob = cv.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))

        net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        net.setInput(blob)
        preds = net.forward()

        idxs = np.argsort(preds[0])[::-1][:5]

        for (i, idx) in enumerate(idxs):
            if i == 0:
                text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
                if "cat" in text:
                    clone = resized.copy()
                    cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
                    cv.putText(clone, "CAT", (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv.rectangle(window, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
                    cv.putText(window, "CAT", (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv.rectangle(imgCrop, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
                    cv.putText(imgCrop, "CAT", (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif "dog" in text:
                    clone = resized.copy()
                    cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 255), 2)
                    cv.putText(clone, "DOG", (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv.rectangle(window, (x, y), (x + winW, y + winH), (0, 255, 255), 2)
                    cv.putText(window, "DOG", (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv.rectangle(imgCrop, (x, y), (x + winW, y + winH), (0, 255, 255), 2)
                    cv.putText(imgCrop, "DOG", (5, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        clone = resized.copy()
        cv.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv.imshow("Window", clone)
        cv.waitKey(1)
    imgCrop = clone.copy()

cv.imshow("FinalOutputImage", imgCrop)
cv.imwrite("output.jpg", imgCrop)

cv.waitKey(0)
cv.destroyAllWindows()
