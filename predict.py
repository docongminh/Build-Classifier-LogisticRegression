import cv2
import pickle
import numpy as np 
import matplotlib.pyplot as plt 
import argparse
from preprocess_image import load_image, fit_Kmeans_hog 
from setup_model import LogisticRegression 

# setup argument

arg = argparse.ArgumentParser()

arg.add_argument("-m", "--model", required = True, help = " path to model")
arg.add_argument("-i", "--image", required = True, help = " path to image for predict")
args = vars(arg.parse_args())

list_images = load_image(args["image"])
images = fit_Kmeans_hog(list_images, 5)

model = pickle.load(open(args["model"], 'rb'))

for idx, img in enumerate(images):
	plt.imshow(list_images[idx])
	pre = model.predict(img.reshape(-1, 350*350))
	if pre[0] == 1:
		plt.title("Metropolitian")
	else:
		plt.title("Countryside")
	plt.show()
