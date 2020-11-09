import numpy as np
import cv2
from keras.models import load_model

###################################
width = 640
height =480
threshold = 0.00
###################################
cap = cv2.VideoCapture(1)
cap.set(3,width)
cap.set(4,height)

new_model = load_model('name of trained model')

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img) #uniformly distrubutes the lightning in image
    img = img/255
    return img


while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    #cv2.imshow("Processed image",img)
    img = img.reshape(1,32,32,1)
    #Predict
    classIndex =  int(new_model.predict_classes(img))
    #print(classIndex)
    prediction = new_model.predict(img)
    #print(prediction)
    probVal = np.amax(prediction)
    #print(classIndex,probVal)

    if probVal > threshold:
        cv2.putText(imgOriginal,str(classIndex) +" " + str(probVal),(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1,(0,0,255),1)
    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

