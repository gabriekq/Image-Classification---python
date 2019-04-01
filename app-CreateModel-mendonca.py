from tensorflow.python.client import device_lib
import tensorflow as tf


import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm

#Machine learn
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


gpu_name = tf.test.gpu_device_name()
print(gpu_name)

Train_Dir = "Q:/Programacao/data/data_york_dog"

IMG_SIZE = 50
LearnRate = 1e-3

model_name = 'dogs-{}-{}.model'.format(LearnRate, '6conv-basic-video')


def create_train_data():
    training_data = []

    for img in tqdm(os.listdir(Train_Dir)):
        img_num = img.split('.')[0]
        path = os.path.join(Train_Dir,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE) )
        training_data.append([np.array(img),np.array([1])])

    shuffle(training_data)
    return training_data



train_data = create_train_data()
print("Data Ready !!!")


convnet = input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')

convnet = conv_2d(convnet,32,5,activation='relu')
convnet = max_pool_2d(convnet,5)

convnet = conv_2d(convnet,64,5,activation='relu')
convnet = max_pool_2d(convnet,5)

convnet = conv_2d(convnet,32,5,activation='relu')
convnet = max_pool_2d(convnet,5)

convnet = conv_2d(convnet,64,5,activation='relu')
convnet = max_pool_2d(convnet,5)

convnet = conv_2d(convnet,32,5,activation='relu')
convnet = max_pool_2d(convnet,5)

convnet = conv_2d(convnet,64,5,activation='relu')
convnet = max_pool_2d(convnet,5)

convnet = fully_connected(convnet,1024,activation='relu')
convnet = dropout(convnet,0.8)

convnet = fully_connected(convnet,1,activation='softmax')
convnet = regression(convnet,optimizer='adam',learning_rate=LearnRate,loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=4, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True, run_id=model_name)

#model.save(model_name)
model.save('modelMendonca.model')


#1.13.3 numpy


