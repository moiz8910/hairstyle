import pickle
import cv2

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors


import numpy as np
from numpy.linalg import norm





with open('embeddings.pkl', 'rb') as f:
    feature_list = np.array(pickle.load(f))
with open('filenames.pkl', 'rb') as f:
    filenames = np.array(pickle.load(f))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(feature_list.shape)

img=image.load_img('sample/000001.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array,axis=0)
preprocessed_img=preprocess_input(expanded_img_array)
result=model.predict(preprocessed_img).flatten()
normalized_result=result / norm(result)

neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances,indices = neighbors.kneighbors([normalized_result])

print(indices)

# for file in indices [0][1:6]:
for file in indices[0]:
    # print(filenames[file])
    temp_img=cv2.imread(filenames[file])
    # cv2.imshow('output',cv2.resize(temp_img,[512,512]))
    cv2.imshow('output', temp_img)
    cv2.waitKey(0)

