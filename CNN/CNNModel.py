import os
import tensorflow as tf
import numpy as np
import json
import glob
import cv2
import pickle
from sklearn.model_selection import train_test_split
def GenerateModel():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same", input_shape=(48,48,1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) 
    
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) 
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) 
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(1,1),padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(4,4),padding="same"))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=7))
    model.add(tf.keras.layers.Softmax())
    model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(48,48,1))
    model.summary()
    return model
def toGray(x):
    x = np.array(x)
    newX = np.empty(shape=(x.shape[0],x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            newX[i][j] = x[i][j][0]
    return newX



def getData(trainPath, trainTxt):
    labels = []
    f = open(trainTxt)
    labels = json.load(f)
    dirs = os.listdir(trainPath)
    X = []
    Y = []
    for _ in dirs:
        currentLabel = [0]*len(labels)
        currentLabel[labels[_]] = 1
        path = os.path.join(trainPath,_)
        for i in glob.glob(os.path.join(path,"*")):
            image = cv2.imread(i)
            image = toGray(image)
            Y.append(currentLabel)
            X.append(image)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def train(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./modelCNN3',
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                    patience=5, min_lr=0.001)
    tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
    h = model.fit(X_train,Y_train,batch_size=256,epochs=48,validation_data=(X_test,Y_test), callbacks=[reduce_lr,model_checkpoint_callback,tb_callback])
    with open('./trainingHistory3', 'wb') as file_pi:
        pickle.dump(h.history, file_pi)
        file_pi.close()
X, Y = getData('./train','./labels.txt')
model = GenerateModel()
train(model, X, Y)

