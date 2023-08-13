
import random, pickle, hdf5storage, itertools, scipy.io as sio, array, numpy as np, time
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
import scipy.io as sio
import tensorflow as tf
import hdf5storage
from tensorflow.keras import losses,regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Reshape,Dense,Dropout,Activation,Flatten, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import matplotlib.pyplot as plt


# ### Load dataset and create labels ###


# %% Initialization, load mat file and create labels
classes = ['LTE','NR','Overlap']
in_dim = [2,880]
num_classes=len(classes)

# load dataset and convert labels to keras compatible
load_data = hdf5storage.loadmat('dataset_snr.mat')
X = load_data['data']
X_label = load_data['label']
load_data1 = sio.loadmat('data_label_snr.mat')
lbl = load_data1['data_label_snr']
X_labeld = tf.keras.utils.to_categorical(X_label,num_classes)
[r1,c1]=X.shape
X_labeld = np.reshape(X_labeld,(r1 ,num_classes))
X=np.reshape(X,(r1,in_dim[0],in_dim[1]))




X.shape


snr=[-15,-10,-5,0,5,10,15,20,25,30]


#### Create training and validation datasets ###


np.random.seed(seed=int(time.time()))
n_examples = X.shape[0]
n_train = n_examples * 0.7
train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)
val_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_val =  X[val_idx]
X_train_label=X_labeld[train_idx]
X_val_label=X_labeld[val_idx]



X_train.shape


# ### Model definition ###

dr = 0.30
model = Sequential()
model.add(Reshape(in_dim+[1],input_shape=in_dim))

# 1st convolution layer
model.add(ZeroPadding2D((2, 2)))
model.add(Conv2D(64, (1, 3), name='conv1', padding="valid", activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd convolutional layer
model.add(ZeroPadding2D((2, 2)))
model.add(Conv2D(32, (2, 3), name='conv2', padding="valid", activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))

# 3rd convolutional layer
model.add(ZeroPadding2D((2, 2)))
model.add(Conv2D(16, (2, 2), name='conv3', activation='relu',  padding="valid", kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))

# 1st Fully connected layer
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer="he_normal", name="dense1",kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dr))

# 2nd Fully connected layer
model.add(Dense(50, activation='relu', kernel_initializer="he_normal", name="dense2",kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(dr))

#Output
model.add(Dense(len(classes), activation='softmax' ))

#Compilie
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()


# ### Model fitting ###

# In[ ]:


#%% Model fitting and evaluation
nb_epoch = 2000
batch_size = 512

filepath = 'LTE_NR_OVERLAP_snr_mapmin_FFT_v1.h5'
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'loss', min_delta = 0, patience = 20, verbose = 0, mode = 'auto')
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=0.00001, cooldown=0, min_lr=0.0001)

history = model.fit(X_train, X_train_label,
                    batch_size=batch_size,
                    epochs=nb_epoch,
                    verbose=1,
                    validation_data=(X_val, X_val_label),
                    callbacks=[early_stop, checkpoint, reduce_lr_callback])
model.load_weights(filepath)
score = model.evaluate(X_val, X_val_label, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




acc = {}
conf_tot={}
for snrs in snr:
    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], val_idx))
    temp=np.where(np.array(test_SNRs)==snrs)
    test_X_i = X_val[temp[0]]
    test_Y_i = X_val_label[temp[0]]

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    conf_tot[snrs]=confnorm
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snrs] = 1.0*cor/(cor+ncor)



import numpy as np
import matplotlib.pyplot as plt

# Assuming you have defined snr, lbl, val_idx, X_val, X_val_label, classes, and model

acc = {}
conf_tot = {}
accuracy_list = []

for snrs in snr:
    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], val_idx))
    temp = np.where(np.array(test_SNRs) == snrs)
    test_X_i = X_val[temp[0]]
    test_Y_i = X_val_label[temp[0]]

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    conf_tot[snrs] = confnorm
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor + ncor))
    acc[snrs] = 1.0 * cor / (cor + ncor)
    accuracy_list.append(cor / (cor + ncor))

# Plot SNR values vs. accuracy
plt.figure(figsize=(10, 6))
plt.plot(snr, accuracy_list, marker='o')
plt.title('SNR Values vs. Accuracy')
plt.xlabel('SNR (Signal-to-Noise Ratio)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


