import math
import cv2
import numpy as np

#Web Camera
cap=cv2.VideoCapture('video.mp4')
count_line_postion=550
min_rect_width=50
min_rect_height=50
detect=[]
list1=[]
a = np.array([0],np.uint8)
offset=6
counter=0
images = []

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

while True:
    ret,frame1=cap.read()
    frame1=cv2.resize(frame1,(1250,720))
    frame_gray=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) 
    images.append(frame_gray)
    # removing the images after every 50 image
    if len(images)==50:
            images.pop(0)
    '''
    Mean Filter:
        In this method background is estimated by taking mean of the previous N frames. 
        Once background is estimated, foreground is estimated by the difference of background and current frame.
    '''

    image = np.array(images)
    image = np.mean(image,axis=0)
    image = image.astype(np.uint8)
    foreground_image = cv2.absdiff(frame_gray,image)
    img_sub = np.where(foreground_image>70,frame_gray,a)
    cv2.imshow('foreground',img_sub)

    dilate=cv2.dilate(img_sub,np.ones((5,5)))#Dilate the blur image
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))#gettting the kernel
    opening=cv2.morphologyEx(dilate,cv2.MORPH_OPEN,np.ones((2,2)))#morphology
    dilatada=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)#remove noise inside the cars
    cv2.imshow('frame',dilatada)
    
    contourShape,hirerchy=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame1,(0,count_line_postion),(1500,count_line_postion),(255,127,0),3)

    for (i,c) in enumerate(contourShape):
        (x,y,w,h)=cv2.boundingRect(c)
        validate_counter=(w>=min_rect_width) and (h>=min_rect_height)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,'Vehicle'+str(counter),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        center=center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,2,(0,0,255),5)

        for (x,y) in detect:
            if y<(count_line_postion+offset) and y>(count_line_postion-offset):
                counter=counter+1
                cv2.line(frame1,(0,count_line_postion),(1500,count_line_postion),(255,0,255),3)
                detect.remove((x,y))

    cv2.putText(frame1,'Vehicle Counter:'+str(counter),(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,127,255),3)  
    cv2.imshow('Video Orginal',frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
cap.release()
