import tensorflow as tf
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

#Machine learn
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

gpu_name = tf.test.gpu_device_name()
print(gpu_name)

train_dir = "Q:/Programacao/data/train-2"
IMG_SIZE = 50
learn_rate = 1e-3
model_name = 'model2.model'
save_model_path = 'Q:/Programacao/data/models/'

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'Donald': return [1,0]
    elif word_label == 'Obama': return [0,1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(train_dir)):
        label = label_img(img)
        img_path = os.path.join(train_dir,img)
        img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    return training_data


train_data = create_train_data()

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

convnet = fully_connected(convnet,2,activation='softmax')

#loss='categorical_crossentropy'

convnet = regression(convnet,optimizer='adam',learning_rate=learn_rate,loss='categorical_crossentropy',name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=4, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True, run_id=model_name)
model.save(save_model_path+model_name)

