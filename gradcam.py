import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import load_model
tf.compat.v1.disable_eager_execution()
import argparse


parser = argparse.ArgumentParser(description='GradCAM')
parser.add_argument('--base', type=str, default='incepv3', help='base model architecture')
parser.add_argument('--model_path', type=str, default='weights.h5', help='path to saved weights')
parser.add_argument('--input_size', type=int, default=299, help='input size')
parser.add_argument('--img', type=str, default='./gradcam_cancer_input.png', help='path to image')
parser.add_argument('--layer', type=str, default='mixed10', help='layer to compute gradients')
parser.add_argument('--out', type=str, default='./gradcam_output.png', help='result directory')
args = parser.parse_args()

base = args.base
SIZE = args.input_size
path = args.img
out_dir = args.out
layer_name = args.layer
model = load_model(args.model_path)


def process(path, base, SIZE):
    x = keras.utils.load_img(path, target_size=(SIZE, SIZE))
    x = keras.utils.img_to_array(x)
    if base in ['vgg16', 'vgg19', 'incepv3', 'xcep']:
        x = x / 255.0
    elif base in ['res50', 'res101']:
        x -= np.mean(x, axis=(0, 1, 2))
        x = x[..., ::-1]
    elif base in ['dense', 'mob', 'incepres']:
        x = (x / 127.5) - 1.0
    else:    
        raise ValueError('Invalid model specified')
    x = np.expand_dims(x, axis=0)

    return x


def main(path, base, SIZE, model, layer_name):
    
    labels = {0: 'Non-cancer pneumonia', 1: 'Cancer'}
    truth = 1 # index of class 'Cancer'
    x_ = keras.utils.load_img(path)
    x_ = keras.utils.img_to_array(x_)
    x = process(path, base, SIZE)

    pred = model.predict(x)
    pred_index = np.argmax(pred)
    pred_label = labels[pred_index]
    print(f'Ground truth: {truth} - {labels[truth]}')
    print(f'Predicted: {pred_index} - {pred_label}')
    
    output_mod = model.output
    output_all_cats = output_mod[0] 
    y_c = output_all_cats[pred_index] 
    spatial_map_layer = model.get_layer(layer_name).output

    grads_l = K.gradients(y_c, spatial_map_layer)
    grads = grads_l[0]
    spatial_map_and_gradient_function = K.function([model.input], [spatial_map_layer, grads])
    spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([x])
 
    spatial_map_val = spatial_map_all_dims[0]
    grads_val = grads_val_all_dims[0]
    weights = np.mean(grads_val, axis=(0,1))
    
    cam = np.dot(spatial_map_val, weights) 
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (x_.shape[1], x_.shape[0]), cv2.INTER_AREA)
    cam = cam / cam.max()

    plt.imshow(keras.utils.load_img(path), cmap='gray')
    plt.imshow(cam, cmap = 'jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(out_dir, bbox_inches='tight')
    plt.close()
    print(f'GradCAM output saved at {out_dir}')

if __name__ == '__main__':
    main(path, base, SIZE, model, layer_name)

