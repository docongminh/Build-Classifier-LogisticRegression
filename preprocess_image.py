# import necessery lib for ML
import cv2
import glob
import numpy as np
from skimage.feature import hog


def load_image(path_to_image_folder):
    """
        param: path_to_image_folder: folder contain list of images
        return:
            numpy array list of images,and resize to(350,350), shape = (-1,350,350,1) 
    """
    list_image = glob.glob(path_to_image_folder + "*")
    contains_image = []
    for image in list_image:
        img = cv2.imread(image)
        resize_img = cv2.resize(img,(350,350))
        contains_image.append(resize_img)

    return np.array(contains_image)

def label_image(n_classes_1, n_classes_2):
    """
        specify: n_classes_1 is metrolitian and n_classes_2 is country
        parames: n_classes_1, n_classes_2: numbers of images in class_1 and class_2 for labels image

        return: numpy array of label images
    """
    label_1 = np.ones(n_classes_1, dtype = int)
    label_2 = np.zeros(n_n_classes_2, dtype = int)

    return np.append(label_1, label_2)

def setUp_kmeans(image, k):
    """
        params: image: image for clustering
                k: number of centroid
        return: numpy array of image clustered
    """
    reshape_image = image.reshape((-1, 3))

    reshape_image = np.float32(reshape_image)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, centroid = cv2.kmeans(reshape_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centroid = np.uint8(centroid)
    res = centroid[label.flatten()]
    res2 = res.reshape((image.shape))

    return res2

def fit_Kmeans_hog(list_image, k):
    """
    params: list_image: list of image contain images
            k: number of centroid for K-means
            notice:use k = 5 for extract feature that help reduce overfiting
    return: numpy array of image clustered that data for training
    """
    temp_array = []
    for image in list_image:
        out_img = setUp_kmeans(image, k)
        feature, hog_image = hog(out_img, orientations = 10, pixels_per_cell = (4,4), cells_per_block = (2,2), visualize = True)
        temp_array.append(hog_image)

    return np.array(temp_array)

    


