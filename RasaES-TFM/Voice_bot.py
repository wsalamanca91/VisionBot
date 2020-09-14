## Run this command in terminal  before executing this program
## rasa run -m models --endpoints endpoints.yml --port 5002 --credentials credentials.yml
## and also run this in seperate terminal
## rasa run actions

import requests
import sys
import speech_recognition as sr     # import the library
# import subprocess
# from gtts import gTTS

import pyttsx3
engine = pyttsx3.init()
# These will be system specific
es_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-ES_HELENA_11.0"
engine.setProperty('voice', es_voice_id)

# sender = input("What is your name?\n")

bot_message = ""
message=""

r = requests.post('http://localhost:5002/webhooks/rest/webhook', json={"message": "Hola"})

print("Visual bot dice, ",end=' ')
for i in r.json():
    bot_message = i['text']
    print(f"{bot_message}")


engine.say(bot_message)
engine.runAndWait()

# myobj = gTTS(text=bot_message)
# myobj.save("welcome.mp3")
# print('saved')
# Playing the converted file
# subprocess.call(['mpg321', "welcome.mp3", '--play-and-exit'])

while 'adiós' not in message and 'gracias' not in message and 'hasta luego'not in message and 'nos vemos' not in message:
    r = sr.Recognizer()  # initialize recognizer
    with sr.Microphone() as source:  # mention source it will be either Microphone or audio files.
        print("Di algo :")
        audio = r.listen(source)  # listen to the source
        try:
            message = r.recognize_google(audio, language = 'es-ES')  # use recognizer to convert our audio into text part.
            print("Tú dices : {}".format(message))

        except:
            print("Lo siento, no he logrado reconocer lo que dices")  # In case of voice not recognized  clearly
    if len(message)==0:
        continue
    #print("Enviando mensaje ahora...")

    r = requests.post('http://localhost:5002/webhooks/rest/webhook', json={"message": message})

    print("Visual bot dice, ",end=' ')
    
    for i in r.json():
        bot_message = i['text']
        print(f"{bot_message}")

    engine.say(bot_message)
    engine.runAndWait()
    
    # myobj = gTTS(text=bot_message)
    # myobj.save("welcome.mp3")
    # print('saved')
    # # Playing the converted file
    # subprocess.call(['mpg321', "welcome.mp3", '--play-and-exit'])

