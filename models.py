import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import ResNet50, ResNet101, VGG16, VGG19, InceptionV3, MobileNetV2, InceptionResNetV2, Xception
from keras.applications.densenet import DenseNet121
tf.compat.v1.disable_eager_execution()


def densenet121(fine_tune=128, IMAGE_SIZE):
    conv_base = DenseNet121(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_ch))
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    x = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_base.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x) 
    x = tf.keras.layers.Dropout(0.5)(x)
    out_layer = tf.keras.layers.Dense(2, activation='softmax', name='actiation')(x)

    model = Model(inputs=conv_base.input, outputs=out_layer)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=6e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-2)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def mobnetv2(fine_tune=46, IMAGE_SIZE):
    conv_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_ch))
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    x = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_base.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out_layer = tf.keras.layers.Dense(2, activation='softmax', name='actiation')(x)

    model = Model(inputs=conv_base.input, outputs=out_layer)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=5e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def inceptionv3(fine_tune=94, IMAGE_SIZE):
    conv_base = InceptionV3(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_ch))
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    x = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_base.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out_layer = tf.keras.layers.Dense(2, activation='softmax', name='actiation')(x)

    model = Model(inputs=conv_base.input, outputs=out_layer)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-2)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def incepresv2(fine_tune=170, IMAGE_SIZE):
    conv_base = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_ch))
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    x = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_base.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x) 
    x = tf.keras.layers.Dropout(0.5)(x)
    out_layer = tf.keras.layers.Dense(2, activation='softmax', name='actiation')(x)

    model = Model(inputs=conv_base.input, outputs=out_layer)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=7.5e-2)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def resnet50(fine_tune=53, IMAGE_SIZE):
    conv_base = ResNet50(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_ch))
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    x = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_base.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out_layer = tf.keras.layers.Dense(2, activation='softmax', name='actiation')(x)

    model = Model(inputs=conv_base.input, outputs=out_layer)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-2)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def resnet101(fine_tune=104, IMAGE_SIZE):
    conv_base = ResNet101(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_ch))
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    x = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_base.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out_layer = tf.keras.layers.Dense(2, activation='softmax', name='actiation')(x)

    model = Model(inputs=conv_base.input, outputs=out_layer)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-2)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def Vgg16(fine_tune=7, IMAGE_SIZE):
    conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False
    x = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_base.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x) 
    x = tf.keras.layers.BatchNormalization()(x)
    out_layer = tf.keras.layers.Dense(2, activation='softmax', name='actiation')(x)

    model = Model(inputs=conv_base.input, outputs=out_layer)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-6, beta_1=0.9, beta_2=0.999, epsilon=1e-2)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def Vgg19(fine_tune=8, IMAGE_SIZE):
    conv_base = VGG19(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_ch))
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    x = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_base.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x) 
    x = tf.keras.layers.BatchNormalization()(x)
    out_layer = tf.keras.layers.Dense(2, activation='softmax', name='actiation')(x)

    model = Model(inputs=conv_base.input, outputs=out_layer)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-6, beta_1=0.9, beta_2=0.999, epsilon=5e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def xception(fine_tune=41, IMAGE_SIZE):
    conv_base = Xception(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, N_ch))
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
        for layer in conv_base.layers[-fine_tune:]:
            layer.trainable = True
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    x = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_base.output)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x) 
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out_layer = tf.keras.layers.Dense(2, activation='softmax', name='actiation')(x)

    model = Model(inputs=conv_base.input, outputs=out_layer)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

