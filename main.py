import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import cm
from tqdm import tqdm
import cv2
import os
import re
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications import ResNet50, ResNet101, VGG16, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception
from keras.applications.densenet import DenseNet121
from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
tf.compat.v1.disable_eager_execution()
from models import Vgg16, Vgg19, resnet50, resnet101, densenet121, mobnetv2, incepresv2, inceptionv3, xception
from processing import data_processing, get_model
import argparse

SEED = 99
keras.utils.set_random_seed(SEED)

parser = argparse.ArgumentParser(description='Training script for deep learning models')
parser.add_argument('--base', type=str, default='incepv3', help='base model architecture')
parser.add_argument('--input_size', type=int, default=299, help='input size')
parser.add_argument('--data', type=str, default='./sample_data', help='path to data directory')
parser.add_argument('--epochs', type=int, default=120, help='number of epochs')
parser.add_argument('--batch', type=int, default=16, help='batch size')
parser.add_argument('--out', type=str, default='./', help='result directory')

args = parser.parse_args()
data = args.data
IMAGE_SIZE = args.input_size
BATCH_SIZE = args.batch
EPOCHS = args.epochs
base = args.base
out_dir = args.out

X_train, X_val, X_test, Y_train, Y_val, Y_test = data_processing(data, base, IMAGE_SIZE)
print('Validation Shape: ', X_val.shape)
print('Train Shape: ', X_train.shape)
print('Test Shape: ', X_test.shape)

train_datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, 
                                   zoom_range=0.2, shear_range=0.15,
                                   horizontal_flip=True, vertical_flip=True, 
                                   rotation_range=10, fill_mode="nearest")
train_datagen.fit(X_train)
validation_datagen = ImageDataGenerator()
validation_datagen.fit(X_val)
test_datagen = ImageDataGenerator()
test_datagen.fit(X_test)


model = get_model(base)
model.summary(show_trainable=True)


checkpoint_callback = ModelCheckpoint(filepath='weights.h5',
                                      monitor='val_accuracy',
                                      mode='max',
                                      save_best_only=True)
hist = model.fit(train_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                 steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                 epochs=EPOCHS, verbose=2,
                 validation_data=(X_val, Y_val),
                 callbacks=[checkpoint_callback]
                 )

final_loss, final_accuracy = model.evaluate(X_val, Y_val)
print(f'Final Loss: {final_loss}, Final Accuracy: {final_accuracy}')

# Accuracy plot
acc_dir = os.path.join(out_dir, f'{base}_acc_{EPOCHS}e_{IMAGE_SIZE}.png')
loss_dir = os.path.join(out_dir, f'{base}_loss_{EPOCHS}e_{IMAGE_SIZE}.png')
figure(figsize=(6, 4))
plt.plot(hist.history['accuracy'], color="#1338BE")
plt.plot(hist.history['val_accuracy'], color="#B90E0A")
plt.title(f"Accuracy {base}", size=16)
plt.ylabel('Accuracy', size=14)
plt.xlabel('Epoch', size=14)
plt.legend(['Train', 'Validation'], loc='lower right', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(acc_dir,bbox_inches='tight')
plt.close()
print(f"Training and validation accuracy curves saved to {acc_dir}")

# Loss plot
plt.plot(hist.history['loss'], color="#1338BE")
plt.plot(hist.history['val_loss'], color="#B90E0A")
plt.title(f"Loss {base}", size=16)
plt.ylabel('Loss', size=14)
plt.xlabel('Epoch', size=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(['Train', 'Validation'], loc='upper right', fontsize=14)
#plt.ylim([0.05, 0.9])
plt.savefig(loss_dir,bbox_inches='tight')
plt.close()
print(f"Training and validation loss curves saved to {loss_dir}")

mod_dir = os.path.join(out_dir, f"{base}_{EPOCHS}e_{IMAGE_SIZE}.h5")
model.save(mod_dir)
print(f'Training {base} completed. Model saved to {mod_dir}')

print('Evaluating on the testing set')
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(Y_true, Y_pred)
print(cm)
print(confusion_matrix(Y_true, Y_pred))
print(classification_report(Y_true, Y_pred))

# Calculating metrics
total=sum(sum(cm))
accuracy=(cm[0,0]+cm[1,1])/total
print ('Accuracy : ', accuracy)
sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
print('Sensitivity : ', sensitivity )
specificity = cm[0,0]/(cm[0,0]+cm[0,1])
print('Specificity : ', specificity)

# Calculating AUC
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred[:, 1])
auc_ = auc(fpr, tpr)
auc_dir = os.path.join(out_dir, f"{base}_ROC_{EPOCHS}e_{IMAGE_SIZE}.png")

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label=f'{model} = {auc}')
plt.ylabel('True positive rate', size=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('ROC curve', size=16)
plt.legend(loc='best', fontsize=14)
plt.savefig(auc_dir, bbox_inches='tight')
plt.close()
print(f'ROC saved at {auc_dir}')
print('Evaluating completed')
