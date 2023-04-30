import cv2
import numpy as np
cap = cv2.VideoCapture(0)
img1 = cv2.imread('img1.jpg')
whT = 320
confThreshold =0.5
nmsThreshold= 0.2

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
# print(classNames)

modelConfiguration = 'yolov3.cfg'
modelWeights='yolov3.weights'

#network
net=cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

'''
NOTE: 
    - To use readNetFromDarknet, you need to provide the function with two files: 
    a model configuration file (usually with a .cfg extension) 
    and a model weights file (usually with a .weights extension). 
    
    - The configuration file contains information about the structure of the neural network, 
    such as the number and types of layers, 
    while the weights file contains the learned parameters of the network.
    
    - When you call readNetFromDarknet with the paths to these files, 
    it reads the files and creates a network object that can be used for inference.
'''


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)


while True:
    choice = input("Do you want to use your camera or upload an image? Type 'camera' or 'upload': ")
    if choice.lower() == 'camera':
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0],1, crop=False)
            net.setInput(blob)

            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i-1]) for i in net.getUnconnectedOutLayers()]

            outputs = net.forward(outputNames)
            findObjects(outputs, img)

            cv2.imshow('Image', img)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        break

    elif choice.lower() == 'upload':
        img_path = input("Enter the path of the image you want to upload: ")
        img = cv2.imread(img_path)
        blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0],1, crop=False)
        net.setInput(blob)

        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i-1]) for i in net.getUnconnectedOutLayers()]

    '''
    NOTE:
    - blobFromImage is a function in the OpenCV library 
    that preprocesses an input image or frame 
    for use in a deep neural network. 
    
    - It takes an image as input, along with several optional parameters, 
    and returns a preprocessed image in the form of a 
    4-dimensional numpy array or "blob".
    
    - Parameters:
        1/255: the scale factor for the image pixel values. 
        In this case, the pixel values are divided by 255 
        to bring them into the range of 0 to 1, 
        which is often used as input for neural networks.
    
    '''

    layersNames = net.getLayerNames()
    # print(layersNames)

    # print(net.getUnconnectedOutLayers())
    outputNames = [(layersNames[i-1]) for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    # findObjects(outputs, img)

    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv2.imshow('Image', img)

    # cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break

