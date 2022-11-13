import cv2

iname ='E:/My file/File in here/archive/concrete-cracks.jpg'

img =cv2.imread(iname)

print(img.shape)

#to display at jupyter notebook
import matplotlib.pyplot as plt
#Note cv2 read BGR as default
cv2.imshow('test', img)