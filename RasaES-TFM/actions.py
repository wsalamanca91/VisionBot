# This files contains your custom actions which can be used to run
# custom Python code.
#"C:/Users/Beatriz Zaragoza/anaconda3/Scripts/activate"
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"


from typing import Any, Text, Dict, List


from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd

boxes_imagen=[[[1, 2, 3, 4], 'gato'], 
[[2, 3, 4, 5], 'persona'],
[[42.36198, 70.70378, 66.62063, 116.57681], 'persona'],
 [[67.42545, 76.04137, 88.641685, 113.72993], 'persona'],
 [[285.40536, 17.997717, 321.75845, 130.5938], 'persona'],
 [[5.40536, 17.997717, 21.75845, 130.5938], 'perro'],
 [[285.40536, 120, 321.75845, 130.5938], 'coche'],
 [[5.40536, 127.997717, 21.75845, 130.5938], 'señal de trafico'],
 [[65.40536, 127.997717, 121.75845, 130.5938], 'señal de trafico'],
 [[65.40536, 7.997717, 121.75845, 13.5938], 'ave'],
 [[255.40536, 7.997717, 301.75845, 13.5938], 'ave']]

def getItems(item, value):
    if(item == None):
        item = list()
        item.append(value)
        return item
    item.append(value)
    return item
    

x_max = 322
y_max = 156
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
        #images2[posicion] = [[x,getItems(item,boxes_imagen[i][1]).count(x)] for x in set(getItems(item,boxes_imagen[i][1]))]
        
        #dict2[i] =[[x,getItems(item,boxes_imagen[i][1]).count(x)] for x in set(getItems(item,boxes_imagen[i][1]))]
        #vector_pos_label.append([posicion,boxes_imagen[i][1],1])

    # pos_label=pd.DataFrame(vector_pos_label,columns=['Posicion','Labels','Cuenta'])
    # group_pos_label=pos_label.groupby(['Posicion','Labels']).count()
    # group_pos_label.reset_index(level=1,inplace=True)
    # Diccionario=group_pos_label.transpose().to_dict('list')
    return images



posicion_objeto = parametro_a_diccionario(x_max,y_max,boxes_imagen)
def cuenta_dic(diccion):
    dict2={}
    for i in diccion.keys():
        dict2[i] =[[x,diccion[i].count(x)] for x in set(diccion[i])]
    return dict2

posicion_objeto= cuenta_dic(posicion_objeto)
print(posicion_objeto)



#'Arriba a la Izquierda': ['person', 1],
#posicion_objeto = { 'Delante': ['person', 1], 'Derecha': ['person', 1], 'Izquierda': ['person', 1]}

# posicion_objeto = {
#     'arriba a la izquierda': ['person', 1],
#     'delante': ['person', 1],
#     'derecha': ['person', 1],
#     'izquierda': ['person', 1]
#     }

class Action_Objeto_Posicion(Action):

     def name(self) -> Text:
         return "action_objeto_posicion"

     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        posicion = tracker.get_slot("posicion")
        posicion_objetos = posicion_objeto.get(posicion)

        if posicion_objetos is None:
            output = "La posición {} no se encuentra en las posibilidades".format(posicion)
        else:
            a = 0
            for i in posicion_objetos:  
                
                if posicion=="derecha" or posicion=="izquierda":
                    output0 = "A la {} encontramos {} {}".format(posicion, i[1], i[0] )
                else:
                    output0 = "{} encontramos {} {}".format(posicion, i[1], i[0] )
                    
                if a == 0:
                    output = output0
                else:
                    output = output +'\n' + output0
                a = a+1
                
        dispatcher.utter_message(text=posicion)

        return []


