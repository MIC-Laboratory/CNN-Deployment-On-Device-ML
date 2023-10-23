import serial
import time
import numpy as np
import cv2
# Define the serial port and baud rate
serial_port = 'COM4'
baud_rate = 1152000

# Create a serial object
ser = serial.Serial(serial_port, baud_rate)
data = ser.readline()
size = 0
image_array = []
start_flag = False
size = 160, 120
duration = 2
fps = 25
start = time.time()
output_index = 0
classes = ["Black","Green","Red","Yellow"]
while not data.startswith(b"END"):
    
    if (data.startswith(b"SIZE")):
        size = data.decode().strip().split(":")[1]
    elif (data.startswith(b"START")):
        start_flag = True
    elif (data.startswith(b"output")):
        output_index = int(data.decode().strip().split(":")[1])
        
    elif start_flag:
        # image_array.append(data)
        
        image_array = np.frombuffer(data,dtype=np.uint8)
        
        # decoded_img = cv2.imdecode(image_array,)
        decoded_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        decoded_img = cv2.resize(decoded_img,(640,480))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (decoded_img.shape[1]-200,decoded_img.shape[0]-20)
        color = (255, 255, 255)
        thickness = 2
        fontScale = 1
        cv2.putText(decoded_img, classes[output_index], org, font, fontScale, color, thickness)
        cv2.imshow('image', decoded_img)
        cv2.waitKey(1)
        # print(start - time.time())
        start = time.time()
        start_flag=False
        output_index = 0

    if start_flag:
        
        data = ser.read(int(size))
        
    else:
        data = ser.readline()
        # print(data)

cv2.destroyAllWindows()
ser.close()
