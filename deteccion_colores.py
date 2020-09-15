from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76, deltaE_ciede2000
import os

class deteccion_colores():
    def __init__(self): 

        self.BDD_COLORES = {
        'azul': [240,248,255],
        'lavanda': [230,230,250],
        'azul pálido': [176,224,230],
        'azul claro': [173,216,230],
        'azul cielo': [135,206,250],
        'azul cielo': [135,206,235],
        'azul cielo': [0,191,255],
        'acero claro': [176,196,222],
        'azul': [30,144,255],
        'azul': [100,149,237],
        'azul': [70,130,180],
        'azul': [95,158,160],
        'morado': [123,104,238],
        'lila': [106,90,205],
        'morado oscuro': [72,61,139],
        'azul pavo': [65,105,225],
        'azul': [0,0,255],
        'azul eléctrico': [0,0,205],
        'azul oscuro': [0,0,139],
        'NAVY': [0,0,128],
        'azul medianoche': [25,25,112],
        'azul violaceo': [138,43,226],
        'índigo': [75,0,130],
        'oro claro': [250,250,210],
        'oro': [238,232,170],
        'oro': [240,230,140],
        'oro': [218,165,32],
        'oro': [255,215,0],
        'naranja': [255,165,0],
        'naranja': [255,140,0],
        'PERU': [205,133,63],
        'CHOCOLATE': [210,105,30],
        'marrón': [139,69,19],
        'SIENNA': [160,82,45],
        'oro': [255,223,0],
        'oro': [212,175,55],
        'oro': [207,181,59],
        'oro': [197,179,88],
        'oro': [230,190,138],
        'oro': [153,101,21],
        'SALMON': [255,160,122],
        'SALMON': [250,128,114],
        'SALMON': [233,150,122],
        'CORAL': [240,128,128],
        'rosa': [205,92,92],
        'rosa fucsia': [220,20,60],
        'rosa': [178,34,34],
        'rosa': [255,0,0],
        'rosa oscuro': [139,0,0],
        'rosa oscuro': [128,0,0],
        'tomate': [255,99,71],
        'naranja rosa': [255,69,0],
        'rosa': [219,112,147],
        'verde': [124,252,0],
        'verde': [127,255,0],
        'lima': [50,205,50],
        'lima': [0,255,0],
        'verde bosque': [34,139,34],
        'verde oscuro': [0,128,0],
        'verde oscuro': [0,100,0],
        'verde amarillo': [173,255,47],
        'verde amarillo': [154,205,50],
        'verde': [0,255,127],
        'verde primavera': [0,250,154],
        'verde claro': [144,238,144],
        'verde': [152,251,152],
        'verde mar': [143,188,143],
        'verde mar': [60,179,113],
        'verde mar': [32,178,170],
        'verde mar': [46,139,87],
        'verde oliva': [128,128,0],
        'verde oliva': [85,107,47],
        'verde oliva': [107,142,35],
        'CYAN': [224,255,255],
        'CYAN': [0,255,255],
        'agua': [0,255,255],
        'agua marina': [127,255,212],
        'agua marina': [102,205,170],
        'turquesa': [175,238,238],
        'turquesa': [64,224,208],
        'turquesa': [72,209,204],
        'turquesa': [0,206,209],
        'turquesa oscuro': [32,178,170],
        'azul': [95,158,160],
        'cyan oscuro': [0,139,139],
        'cyan': [0,128,128],
        'gris': [220,220,220],
        'gris': [211,211,211],
        'plata': [192,192,192],
        'gris': [169,169,169],
        'gris': [128,128,128],
        'gris': [105,105,105],
        'gris': [119,136,153],
        'gris': [112,128,144],
        'gris': [47,79,79],
        'negro': [0,0,0],
        'CORAL': [255,127,80],
        'tomate': [255,99,71],
        'naranja rosa': [255,69,0],
        'oro': [255,215,0],
        'naranja': [255,165,0],
        'naranja': [255,140,0],
        'rosa': [255,192,203],
        'rosa': [255,182,193],
        'rosa': [255,105,180],
        'rosa oscuro': [255,20,147],
        'rosa': [219,112,147],
        'violeta': [199,21,133],
        'lavanda': [230,230,250],
        'lavanda': [216,191,216],
        'lavanda': [221,160,221],
        'violeta': [238,130,238],
        'violeta': [218,112,214],
        'fucsia': [255,0,255],
        'MAGENTA': [255,0,255],
        'magenta': [186,85,211],
        'purpura': [147,112,219],
        'azul violaceo': [138,43,226],
        'violeta oscuro': [148,0,211],
        'violeta oscuro': [153,50,204],
        'MAGENTA': [139,0,139],
        'púrpura': [128,0,128],
        'índigo': [75,0,130],
        'blanco': [255,255,255],
        'blanco nieve': [255,250,250],
        'blanco': [240,255,240],
        'blanco': [245,255,250],
        'blanco': [240,255,255],
        'azul': [240,248,255],
        'blanco': [248,248,255],
        'blanco roto': [245,245,245],
        'blanco': [255,245,238],
        'BEIGE': [245,245,220],
        'blanco': [253,245,230],
        'blanco': [255,250,240],
        'blanco': [255,255,240],
        'blanco': [250,235,215],
        'lima': [250,240,230],
        'lavanda': [255,240,245],
        'rosa': [255,228,225],
        'blanco': [255,222,173],
        'amarillo': [255,255,224],
        'amarillo': [255,250,205],
        'oro claro': [250,250,210],
        'PAPAYA': [255,239,213],
        'MOCCASIN': [255,228,181],
        'melocotón': [255,218,185],
        'oro': [238,232,170],
        'oro': [240,230,140],
        'oro': [189,183,107],
        'amarillo': [255,255,204],
        'amarillo': [255,255,153],
        'amarillo': [255,255,102],
        'amarillo': [255,255,51],
        'amarillo': [255,255,0],
        'amarillo oscuro': [204,204,0],
        'amarillo oscuro': [153,153,0],
        'amarillo oscuro': [102,102,0],
        'amarillo oscuro': [51,51,0],
        'negro': [0,0,0],
        'gris': [105,105,105],
        'gris': [128,128,128],
        'gris oscuro': [169,169,169],
        'plata': [192,192,192],
        'blanco': [255,248,220],
        'blanco diamante': [255,235,205],
        'beige': [255,228,196],
        'beige': [255,222,173],
        'beige': [245,222,179],
        'marrón': [222,184,135],
        'marrón': [210,180,140],
        'violeta': [188,143,143],
        'marrón': [244,164,96],
        'oro': [218,165,32],
        'PERU': [205,133,63],
        'CHOCOLATE': [210,105,30],
        'marrón': [139,69,19],
        'SIENNA': [160,82,45],
        'marrón': [165,42,42],
        'marrón': [128,0,0]
        }
        self.colores=['azul',
        'lavanda',
        'azul pálido',
        'azul claro',
        'azul cielo',
        'azul cielo',
        'azul cielo',
        'acero claro',
        'azul pálido',
        'azul',
        'azul',
        'azul',
        'morado',
        'lila',
        'morado oscuro',
        'azul pavo',
        'azul',
        'azul eléctrico',
        'azul oscuro',
        'NAVY',
        'azul medianoche',
        'azul violaceo',
        'índigo',
        'oro claro',
        'oro',
        'oro',
        'oro',
        'oro',
        'naranja',
        'naranja',
        'PERU',
        'CHOCOLATE',
        'marrón',
        'SIENNA',
        'oro',
        'oro',
        'oro',
        'oro',
        'oro',
        'oro',
        'SALMON',
        'SALMON',
        'SALMON',
        'CORAL',
        'rosa',
        'rosa fucsia',
        'rosa',
        'rosa',
        'rosa oscuro',
        'rosa oscuro',
        'tomate',
        'naranja rosa',
        'rosa',
        'verde',
        'verde',
        'lima',
        'lima',
        'verde bosque',
        'verde oscuro',
        'verde oscuro',
        'verde amarillo',
        'verde amarillo',
        'verde',
        'verde primavera',
        'verde claro',
        'verde',
        'verde mar',
        'verde mar',
        'verde mar',
        'verde mar',
        'verde oliva',
        'verde oliva',
        'verde oliva',
        'CYAN',
        'CYAN',
        'agua',
        'agua marina',
        'agua marina',
        'turquesa',
        'turquesa',
        'turquesa',
        'turquesa',
        'azul',
        'azul',
        'cyan oscuro',
        'cyan',
        'gris',
        'gris',
        'plata',
        'gris',
        'gris',
        'gris',
        'gris',
        'gris',
        'gris',
        'negro',
        'CORAL',
        'tomate',
        'naranja rosa',
        'oro',
        'naranja',
        'naranja',
        'rosa',
        'rosa',
        'rosa',
        'rosa oscuro',
        'rosa',
        'violeta',
        'lavanda',
        'lavanda',
        'lavanda',
        'violeta',
        'violeta',
        'fucsia',
        'MAGENTA',
        'magenta',
        'purpura',
        'azul violaceo',
        'violeta oscuro',
        'violeta oscuro',
        'magenta',
        'púrpura',
        'índigo',
        'blanco',
        'blanco nieve',
        'blanco',
        'blanco',
        'blanco',
        'azul',
        'blanco',
        'blanco',
        'blanco',
        'BEIGE',
        'blanco',
        'blanco',
        'blanco',
        'blanco',
        'lima',
        'lavanda',
        'rosa',
        'blanco',
        'amarillo',
        'amarillo',
        'oro claro',
        'PAPAYA',
        'MOCCASIN',
        'melocotón',
        'oro',
        'oro',
        'oro',
        'amarillo',
        'amarillo',
        'amarillo',
        'amarillo',
        'amarillo',
        'amarillo oscuro',
        'amarillo oscuro',
        'amarillo oscuro',
        'amarillo oscuro',
        'negro',
        'gris',
        'gris',
        'gris oscuro',
        'plata',
        'blanco',
        'blanco diamante',
        'beige',
        'beige',
        'beige',
        'marrón',
        'marrón',
        'violeta',
        'marrón',
        'oro',
        'PERU',
        'CHOCOLATE',
        'marrón',
        'SIENNA',
        'marrón',
        'marrón'
            ]

    def RGB2HEX(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


    def extrae_colores(self,image_path, number_of_colors, show_chart):
        
        # Obtenemos la imagen usando OpenCV, convirtiéndola a RGB
        image=cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Modificamos el tamaño de la imagen para que tarde menos en extraer los colores
        modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
        # Modificamos la imagen para que sea un array en 2D, que es lo que espera KMeans
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
        
        clf = KMeans(n_clusters = number_of_colors)
        labels = clf.fit_predict(modified_image)
        
        #Cuenta el número de píxeles asociados a cada clúster encontrado
        counts = Counter(labels)

        # Creamos un diccionario con valor=número de píxeles de cada clúster (que realmente es cada color)
        counts = dict(sorted(counts.items()))
        
        #Obtenemos los valores RGB de los clústeres
        center_colors = clf.cluster_centers_
        
        # Ordenamos los colores usando las claves del diccionario counts
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]
        
        if (show_chart):
            plt.figure(figsize = (8, 6))
            plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
            
            
        return rgb_colors,counts

    
    def colores_mayoritarios(self, image,colores =None, threshold = 60, number_of_colors = 6): 
        if colores == None : 
            colores = self.colores
        else:
            pass
        
        
        image_colors,counts = self.extrae_colores(image, number_of_colors, False)

        contador_colores={}
        contador_definitivo={}
        pixeles_mayor=0
        
        #Inicializamos el diccionario de colores a 0 (cada color tendrá asociados 0 píxeles):
        for color in colores:
            contador_colores[color]=0
            
        for i in range(number_of_colors):
            #Cada color extraído de la imagen lo pasamos a formato L*a*b* y guardamos su valor
            curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
            curr_value=counts[i]
            diff_min=threshold
            
            for color in colores:
                selected_color = rgb2lab(np.uint8(np.asarray([[self.BDD_COLORES[color]]])))
                select_image = False
                
                diff = deltaE_ciede2000(selected_color, curr_color)
                
            #   if (diff < threshold):
            # Nos quedamos como color definitivo de todo nuestro listado el que esté a menor
            # distacia del color que estemos evaluando
                if (diff<diff_min):
                    diff_min=diff
                    color_min=color
            
            #El color al que hemos asociado cada color de la extracción suma el número de píxeles de ese color           
            contador_colores[color_min]=contador_colores[color_min]+curr_value
        
        # De toda nuestra base de datos de colores, nos quedamos sólo con los colores que tienen píxeles
        for color in colores:
            if contador_colores[color]!=0:
                contador_definitivo[color]=contador_colores[color]

        return contador_definitivo

    def color_mayoritario(self, image, threshold = 60, number_of_colors = 6): 
        colores = self.colores
        
        image_colors,counts = self.extrae_colores(image, number_of_colors, False)

        contador_colores={}
        contador_definitivo={}
        pixeles_mayor=0
        
        #Inicializamos el diccionario de colores a 0 (cada color tendrá asociados 0 píxeles):
        for color in colores:
            contador_colores[color]=0
            
        for i in range(number_of_colors):
            #Cada color extraído de la imagen lo pasamos a formato L*a*b* y guardamos su valor
            curr_color = rgb2lab(np.uint8(np.asarray([[image_colors[i]]])))
            curr_value=counts[i]
            diff_min=threshold
            
            for color in colores:
                selected_color = rgb2lab(np.uint8(np.asarray([[self.BDD_COLORES[color]]])))
                select_image = False
                
                diff = deltaE_ciede2000(selected_color, curr_color)
                
            #   if (diff < threshold):
            # Nos quedamos como color definitivo de todo nuestro listado el que esté a menor
            # distacia del color que estemos evaluando
                if (diff<diff_min):
                    diff_min=diff
                    color_min=color
            
            #El color al que hemos asociado cada color de la extracción suma el número de píxeles de ese color           
            contador_colores[color_min]=contador_colores[color_min]+curr_value
        
        # De toda nuestra base de datos de colores, nos quedamos sólo con los colores que tienen píxeles
        for color in colores:
            if contador_colores[color]!=0:
                contador_definitivo[color]=contador_colores[color]
                
        #Nos quedamos sólo con el mayoritario
        for color in contador_definitivo:
            if contador_definitivo[color]>pixeles_mayor:
                color_mayoritario_final=color
                pixeles_mayor=contador_definitivo[color]
        
        #return contador_definitivo
        return color_mayoritario_final


