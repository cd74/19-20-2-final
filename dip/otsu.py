import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def otsu(img,w=1):            
	height = img.shape[0]
	width = img.shape[1]
	count = np.zeros(256)
 
	for i in range(height):
		for j in range(width):
			count[int(img[i][j])] += 1 
 
	max_variance = 0.0
	best_thresh = 0
	for thresh in range(1,256):
		n0 = count[:thresh].sum()
		n1 = count[thresh:].sum()
		p0 = n0 / (height * width)
		p1 = n1 / (height * width)
		u0 = 0.0
		u1 = 0.0
		
		for i in range(thresh):
			u0 += i * count[i]/(height * width)/p0
		for j in range(thresh, 256):
			u1 += j * count[j]/(height * width)/p1
		
		if(w==0):
			w = p0
		tmp = w*p0*np.power((u0), 2) + p1 * np.power((u1), 2)
 
		if tmp > max_variance:
			best_thresh = thresh
			max_variance = tmp
 
	return best_thresh

def drawhist(img,f):
	hist = cv2.calcHist([img],[0], None, [256], [0.0,255.0])   #彩色图有三个通道，通道b:0,g:1,r:2
	plt.plot(hist, 'red')  #画图
	plt.savefig('res/h_'+f+'.png')
	plt.cla()

for root,dir,files in os.walk('img/'):
	for f in files:
		img=cv2.imread(root+f,cv2.IMREAD_GRAYSCALE)
		drawhist(img,f)
		th = otsu(img,1)
		print(th)
		ret,dst=cv2.threshold(img,th,255,cv2.THRESH_BINARY)
		ret,dst1=cv2.threshold(img,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
		f1 = f+'.png'
		cv2.imwrite('res/1.0'+f,dst)
		cv2.imwrite('res/otsu_'+f,dst1)