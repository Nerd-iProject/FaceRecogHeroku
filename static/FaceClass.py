import keras
import pickle
import cv2
import numpy as np
import sys

'''This is to supress the tensorflow warnings. If something odd happens, remove them and try to debug form the warnings'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
'''This is to supress the tensorflow warnings. If something odd happens, remove them and try to debug form the warnings'''







face_detection_path= "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
proto_path = "face_detection_model/deploy.prototxt"
model_path = 'pickle/holly_MobileNet_3(50_class).h5'
label_path = 'pickle/holly_50_classes_lableencoder.pickle'



class FaceIndentity:

    def __init__(self, caffe_path, proto_path, model_path, label_path):

        self.detector = cv2.dnn.readNetFromCaffe(proto_path, caffe_path)

        self.model = keras.models.load_model(model_path)

        self.labelencoder = pickle.load(open(label_path,'rb'))



    def predict_image(self, image):
        image_np = np.asarray(image)
        self.getFace_CV2DNN(image)


    def getFace_CV2DNN(self, image):
        facelist = []
        (h,w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)),1.0, (300,300),(104.0, 177.0, 123.0), swapRB= False, crop = False)
        self.detector.setInput(blob)
        detections = self.detector.forward()
        fH = 0
        fW = 0
        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2];

            if confidence < 0.7:
              continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)


            fH = endX - startX
            fW = endY - startY
            if fH < 20 or fW < 20:
              continue
            facelist.append((startX,startY,endX, endY))

        self.setLabel(facelist, image)

    def setLabel(self, facelist,image):
        for (x1,y1,x2,y2) in facelist:

            face = image[y1:y2, x1:x2]
            if(face.shape == (0,0,3)):
                return
            try :
                im = cv2.resize(face, (224, 224)).astype(np.float32) / 255.0
                im = im.reshape(1,224,224,3)
                out = self.model.predict(im)

                label = np.argmax(out)

                name = self.labelencoder.get(label)[5:]
                print('Person Found is :',name)
                cv2.putText(img= image,
                            text=name,
                            org=(x1,y1),
                            fontFace = cv2.FONT_HERSHEY_COMPLEX,
                            fontScale= 0.5,
                            color=(255,100,50),
                            thickness= 1,
                            lineType=cv2.LINE_AA)
            except Exception as e:
                print("Some Error in image: ", e)

reg = FaceIndentity(face_detection_path,proto_path,model_path,label_path)

image = cv2.imread(sys.argv[1])
reg.predict_image(image)
