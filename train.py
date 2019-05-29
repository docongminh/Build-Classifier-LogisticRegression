import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from preprocess_image import load_image, label_image, fit_Kmeans_hog
from setup_model import LogisticRegression
import pickle

 # setup argumen
arg = argparse.ArgumentParser()
arg.add_argument("-hm", "--inputmetro", required=True, help = "path to metro data")
arg.add_argument("-hc","--inputcountry", required=True, help = "path to country data")
args = vars(arg.parse_args())

#load data
metro_data = load_image(args["inputmetro"])
country_data = load_image(args["inputcountry"])

#setup data for training
data = np.append(metro_data, country_data, axis = 0)
data_training = fit_Kmeans_hog(data, 5)

#label image
label = label_image(len(metro_data), len(country_data))

def training(data,
			label,
			learning_rate = 0.001,
			batch_size = 32,
			epochs = 10,
			threshold = 0.5
):

	X_train, X_test, y_train, y_test = train_test_split(data, label, random_state = 1, test_size = 0.2)

	model = LogisticRegression()
	model.fit(X_train, y_train,
			learning_rate = learning_rate,
			batch_size = batch_size,
			threshold = threshold,
			epoch = epochs
			)
	print("Weight", model.W)

	print("Validation", model.evaluate(x_train[split_size:], y_train[split_size:]))

	# evaluate model

	print("evaluate model: ",model.evaluate(X_test, y_tests))

	# save model
	pickle.dump(model, open('./model/model.sav', 'wb'))

if __name__ == '__main__':
	training()



