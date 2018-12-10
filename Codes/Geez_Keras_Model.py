
#from keras.datasets import mnist
from __future__ import print_function
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
batch_size = 128
num_classes = 19
epochs = 50
img_rows= 64
img_cols = 64
num_pixels = img_cols*img_rows

# load data
def shaffle_Normalize(data,labels):
    X_train = data / 255
    s = np.arange(X_train.shape[0])
    np.random.shuffle(s)
    X_train=X_train[s]
    y_train=labels[s]
    return X_train,y_train

def load_data():
    data=np.load('GeezDigitsDataset.npy')
    labels=np.load('One_Hot_labels.npy')
    X_train,y_train=shaffle_Normalize(data,labels)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)
    return X_train,y_train,X_test,y_test

def prepare4training():
    batch_size = 128
    num_classes = 19
    epochs = 10
    X_train,y_train,X_test,y_test=load_data()
    # input image dimensions
    img_rows, img_cols = 64, 64

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train,y_train,X_test,y_test

def builgCNN():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(64,64,3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    BatchNormalization(axis=-1)
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    BatchNormalization(axis=-1)
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    # Fully connected layer

    BatchNormalization()
    model.add(Dense(512))
    model.add(Activation('relu'))
    BatchNormalization()
    model.add(Dropout(0.2))
    model.add(Dense(19))

    model.add(Activation('softmax'))

    return model

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def baseline_model_CNN():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# define the larger model
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def trainingModel(model,X_train,y_train,X_test,y_test,model_num):
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    """history=model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(X_test, y_test))"""
    history = model.fit(X_train, y_train, validation_split=0.33, epochs=epochs, batch_size=batch_size, verbose=2)
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    #fig = plt.figure(figsize=(12,8))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Accuracy_model_'+str(model_num))
    #plt.show()
    # summarize history for loss
    #plt.close(fig)
    #fig = plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Loss_model_'+str(model_num))
    #plt.close(fig)

    with open('history_model_'+str(model_num)+'.json', 'w') as f:
        json.dump(history.history, f)

    model.save('mnist_keras_cnn_model.h5')


    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

X_train,y_train,X_test,y_test=prepare4training()
model=builgCNN()
model1=larger_model()
#trainingModel(model,X_train,y_train,X_test,y_test,1)
trainingModel(model1,X_train,y_train,X_test,y_test,2)
