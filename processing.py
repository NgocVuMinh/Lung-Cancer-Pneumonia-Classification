import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from models import Vgg16, Vgg19, resnet50, resnet101, densenet121, mobnetv2, incepresv2, inceptionv3, xception


def data_processing(data, IMAGE_SIZE):

    disease_types=['Non-cancer','Cancer']
    train_dir = os.path.join(data, 'Training')
    val_dir = os.path.join(data, 'Validation')
    test_dir = os.path.join(data, 'Testing')

    train_data = []
    for id, categ_name in enumerate(disease_types):
        for file in os.listdir(os.path.join(train_dir, categ_name)):
            train_data.append(['{}/{}'.format(categ_name, file), id, categ_name])
    train = pd.DataFrame(train_data, columns=['File', 'DiseaseID','Disease Type'])

    val_data = []
    for id, categ_name in enumerate(disease_types):
        for file in os.listdir(os.path.join(val_dir, categ_name)):
            val_data.append(['{}/{}'.format(categ_name, file), id, categ_name])
    val = pd.DataFrame(val_data, columns=['File', 'DiseaseID','Disease Type'])

    test_data = []
    for id, categ_name in enumerate(disease_types):
        for file in os.listdir(os.path.join(test_dir, categ_name)):
            test_data.append(['{}/{}'.format(categ_name, file), id, categ_name])
    test = pd.DataFrame(test_data, columns=['File', 'DiseaseID','Disease Type'])


    X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
    for i, file in tqdm(enumerate(train['File'].values)):
        image = cv2.imread(os.path.join(train_dir, file))
        if image is not None:
            X_train[i] = cv2.resize(image.copy(), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    X_val = np.zeros((val.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
    for i, file in tqdm(enumerate(val['File'].values)):
        image = cv2.imread(os.path.join(val_dir, file))
        if image is not None:
            X_val[i] = cv2.resize(image.copy(), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    X_test = np.zeros((test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
    for i, file in tqdm(enumerate(test['File'].values)):
        image = cv2.imread(os.path.join(test_dir, file))
        if image is not None:
            X_test[i] = cv2.resize(image.copy(), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    if args.model in ['vgg16', 'vgg19', 'incepv3', 'xcep']:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0
    elif args.model in ['res50', 'res101']:
        X_train -= np.mean(X_train, axis=(0, 1, 2))
        X_train = X_train[..., ::-1]
        X_val -= np.mean(X_val, axis=(0, 1, 2))
        X_val = X_val[..., ::-1]
        X_test -= np.mean(X_test, axis=(0, 1, 2))
        X_test = X_test[..., ::-1]
    elif args.model in ['dense', 'mob', 'incepres']:
        X_train = (X_train / 127.5) - 1.0
        X_val = (X_val / 127.5) - 1.0
        X_test = (X_test / 127.5) - 1.0
    else:    
        raise ValueError('Invalid model specified')
    
    Y_train = to_categorical(train['DiseaseID'], num_classes=2)
    Y_val = to_categorical(val['DiseaseID'], num_classes=2)
    Y_test = to_categorical(test['DiseaseID'], num_classes=2)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def get_model(base):
    d = {
        'dense': densenet121,
        'mob': mobnetv2,
        'incepres': incepresv2,
        'res101': resnet101,
        'res50': resnet50,
        'incepv3': inceptionv3,
        'vgg16': Vgg16, 
        'vgg19': Vgg19, 
        'xcep': xception
        }
    try:
        model = d[base]()
    except KeyError:
        raise ValueError('Invalid model specified')
    return model
