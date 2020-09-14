import requests
import sys
import speech_recognition as sr     # import the library
# import subprocess
# from gtts import gTTS
import time
import cv2
import pyttsx3
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import multiprocessing

import pandas as pd
import numpy as np
from keras.models import load_model
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import base64

import json
from json import JSONEncoder
from PIL import Image

RASA_MODEL_URL = 'http://localhost:5002/webhooks/rest/webhook'
RETINA_NET_MODEL_URL = "/home/wilson/Documentos/TFM/retinaNet/Libraries/resnet50_csv_50.h5"
LABELS_URL = "/home/wilson/Documentos/TFM/VisionBot/files/categorias_catdef_todas.csv"
API_PREDICT = "https://aac8774766a9.ngrok.io/predict"
IMAGE_LOCATION = "/home/wilson/Documentos/TFM/VisionBot/TestImage.jpg"

def predictAPI(image):    
    print(str(type(image)))
    cv2.imwrite( IMAGE_LOCATION, image)
    img = cv2.imread(IMAGE_LOCATION)
    _, img_encoded = cv2.imencode('.jpg', img)
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}
    response = requests.post(API_PREDICT, data=img_encoded.tostring(), headers=headers)
    return json.loads(json.loads(response.text))

def saveLastPredicted(image, boxes, labels):
    COLORS = np.random.uniform(0, 255, size=(len(labels), 3))
    (startX, startY, endX, endY) = boxes[0][0].astype("int")
    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[0], 2)
    cv2.imwrite(IMAGE_LOCATION, image)

def videoProcess(ns, event):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        ns.value = frame
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()

def audioProcess(ns, event):
    #model = loadModel()
    labels_to_names = loadLabels()
    engine = pyttsx3.init()

    es_voice_id = "spanish"
    engine.setProperty('voice', es_voice_id)
    bot_message = ""
    message=""

    while 'adiós' not in message and 'gracias' not in message and 'hasta luego'not in message and 'nos vemos' not in message:

        r = sr.Recognizer()  # initialize recognizer
        with sr.Microphone() as source:  # mention source it will be either Microphone or audio files.
            message = processVoice(r, source, message)  # In case of voice not recognized  clearly
        if len(message)==0:
            continue
        imageToPredict = ns.value
        response = predictAPI(imageToPredict)
        print(response)

        #r = requests.post(RASA_MODEL_URL, json={"message": message, "data":str(response)})
        r = requests.post(RASA_MODEL_URL, json={"message": message})
        objects = parametro_a_diccionario(imageToPredict.shape[0], imageToPredict.shape[1], response)
        print(objects)
        for i in r.json():
            bot_message = i['text']
        answer = createAnswer(objects, bot_message,labels_to_names)
        handleAnswer(engine, answer)


def handleAnswer(engine, bot_message):
    print("Visual bot dice, ",end=' ')
    engine.say(bot_message)
    engine.runAndWait()

def processVoice(r, source, message):
    print("Di algo :")
    audio = r.listen(source)  # listen to the source
    try:
        message = r.recognize_google(audio, language = 'es-ES')  # use recognizer to convert our audio into text part.
        print("Tu dices : {}".format(message))

    except:
        print("Lo siento, no he logrado reconocer lo que dices")  # In case of voice not recognized  clearly
    return message

def loadLabels():
    return pd.read_csv(LABELS_URL, header=None).T.loc[0].to_dict()

def loadModel():
    print("[INFO] loading model...")    
    model = models.load_model(RETINA_NET_MODEL_URL, backbone_name='resnet50')
    model = models.convert_model(model)
    return model

def parametro_a_diccionario(x_max,y_max,boxes_imagen):
    ref_der = x_max*0.8
    ref_izq = x_max*0.2
    ref_arriba = y_max*0.2
    ref_abajo = y_max*0.8
    vector_pos_label=list()

    images = dict()
    images2 = dict()

    for i in range (0,len(boxes_imagen)):
        x_min_box = boxes_imagen[i][0][0]
        y_min_box = boxes_imagen[i][0][1]
        x_max_box = boxes_imagen[i][0][2]
        y_max_box = boxes_imagen[i][0][3]
        y_med = (y_max_box+y_min_box)/2
        x_med = (x_max_box+x_min_box)/2
        if x_med>ref_der:
            pos_x="derecha"
        elif x_med<ref_izq:
            pos_x="izquierda"
        else:
            pos_x="enfrente"

        if y_med>ref_abajo:
            pos_y="abajo"
        elif y_med<ref_arriba:
            pos_y="arriba"
        else:
            pos_y="enfrente"

        if pos_x=="enfrente" and pos_y!="enfrente":
            posicion=pos_y
        elif pos_y=="enfrente" and pos_x!="enfrente":
            posicion=pos_x
        elif pos_y=="enfrente" and pos_x=="enfrente":
            posicion=pos_x
        else:
            posicion=pos_y+" a la "+pos_x
        
        item = images.get(posicion)
        images[posicion] = getItems(item,boxes_imagen[i][1])
    return images

def createAnswer(items, position, labels_to_names):
    answer = "No entiendo tu pregunta"
    objects = items.get(position)
    objectsString = getStringObjects(objects, labels_to_names)
    if position is None:
        output = "La posición {} no se encuentra en las posibilidades".format(posicion)
    if position=="derecha" or position=="izquierda":
        answer = "A la {} encontramos {}".format(position, objectsString)
    else:
        answer = "{} encontramos {}".format(position,objectsString)
    return answer

def getStringObjects(objects, labels_to_names):
    if(objects == None):
        return "nada"
    countObjects = dict((x,objects.count(x)) for x in set(objects))
    stringObjects = ""
    for key in countObjects:
        item = labels_to_names.get(key)
        count = countObjects[key]
        stringObjects = stringObjects + "{} {}".format(count,getPluralOrSingular(item, count))
        print(key, ":", countObjects[key])
    return stringObjects

def getPluralOrSingular(item, count):
    objects = item.split("-")
    if(count > 1):
        return objects[1]
    return objects[0]

def getItems(item, value):
    if(item == None):
        item = list()
        item.append(value)
        return item
    item.append(value)
    return item

def crop_image(img_name,bbox_list):
    img = Image.open(img_name) 
    img_crop = img.crop((bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3])) 
    
    return img_crop

if __name__ == '__main__':

    mgr = multiprocessing.Manager()
    namespace = mgr.Namespace()
    event = multiprocessing.Event()
    p = multiprocessing.Process(
        target=videoProcess,
        args=(namespace, event),
    )
    c = multiprocessing.Process(
        target=audioProcess,
        args=(namespace, event),
    )

    c.start()
    p.start()

    c.join()
    p.join()