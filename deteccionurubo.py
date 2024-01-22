import cv2
import numpy as np
import subprocess
import time
import sys
import re
import requests
from bs4 import BeautifulSoup

from collections import defaultdict
# https://github.com/ultralytics/ultralytics
from ultralytics import YOLO

# Cargar la pagina y encontrar los iframes que contienen los links al video
# https://www.geeksforgeeks.org/implementing-web-scraping-python-beautiful-soup/
url="https://tigocamaras.bridge4digital.com/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
pagina=requests.get(url, headers=headers)
soup=BeautifulSoup(pagina.content,"html.parser")
resultados=soup.find_all("iframe")
links=[]

# Recorrer los links para obtener el atributo src de los links
for resultado in resultados:
    links.append(resultado['src'])
url=links[0]

# Encontrar las variables address y streamid
pagina=requests.get(url, headers=headers)
soup=BeautifulSoup(pagina.content,"html.parser")
pattern_url_base = re.compile(".*var address = '(.*?)';.*", re.DOTALL)
pattern_url_streamid = re.compile(".*var streamid = '(.*?)';.*", re.DOTALL)

# Buscar en el script JS los parametros base y streamid
scripts = soup.find_all('script')
for script in scripts:
   if(pattern_url_base.match(str(script.string))):
       url_base = pattern_url_base.match(script.string)
   if(pattern_url_streamid.match(str(script.string))):
       url_streamid = pattern_url_streamid.match(script.string)
       # Generar la URL completa
       full_url=url_base.groups()[0]+"streams/"+url_streamid.groups()[0]+"/stream.m3u8"

# Iniciar el servidor de video streaming
# https://github.com/bluenviron/mediamtx
p1=subprocess.Popen(['./mediamtx'])
time.sleep(2)
# Iniciar el envio del video al servidor
# https://phoenixnap.com/kb/install-ffmpeg-ubuntu
p2=subprocess.Popen(['ffmpeg','-re','-i',full_url,'-vcodec','copy','-f','rtsp','-rtsp_transport','tcp','rtsp://127.0.0.1:8554/mystream'])
time.sleep(5)

# Cargar el modelo Yolov8
model = YOLO('yolov8s.pt')

# Acceder a la ruta del video obtenido entregado por el servidor RTSP
video_path = "rtsp://127.0.0.1:8554/mystream"
cap = cv2.VideoCapture(video_path)

# Almacenar el historial del tracking
track_history = defaultdict(lambda: [])
totalcount=[]
tid=0

# Obtener video de la camara
while cap.isOpened():
    # Leer frame a frame del video
    success, frame = cap.read()

    if success:
        # Ejecutar YOLOv8 tracking persistentemente en los frames
        results = model.track(frame, persist=True)

        #Evitar error cuando no hay ningun ID
        if results[0].boxes.id != None:
            # Obtener los ID de los objetos y sus cajas
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

        # Agregar ID y boxes al frame
        annotated_frame = results[0].plot()

        # Dibujar los objetos a seguir
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # Puntos del centro
            if len(track) > 30:  # Mantener los objetos en los siguientes X frames
                track.pop(0)
                tid=track_id
                

            # Dibujar las lineas de seguimiento
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=5)
        
        cv2.putText(annotated_frame, str(tid), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2, cv2.LINE_AA)    
        # Mostrar el frame con las notas
        cv2.imshow("Tracking Urubo", annotated_frame)

        key=cv2.waitKey(1)
        if key==ord("Q") or key==ord("q") or key==27:
            p1.terminate()
            p2.terminate()
            break
    else:
        # Terminar si se pierde el video
        p1.terminate()
        p2.terminate()
        break

# Destruir las ventanas creadas
cap.release()
cv2.destroyAllWindows()
