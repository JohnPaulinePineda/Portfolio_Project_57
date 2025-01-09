##################################
# Loading Python Libraries
##################################
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.metrics import PrecisionAtRecall, Recall 
from tensorflow.keras.utils import img_to_array, array_to_img, load_img

from PIL import Image
from glob import glob
import cv2
import random
import math

##################################
# Setting random seed options
# for the analysis
##################################
def set_seed(seed=123):
    np.random.seed(seed) 
    tf.random.set_seed(seed) 
    keras.utils.set_random_seed(seed)
    random.seed(seed)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

##################################
# Defining file paths
##################################
MODELS_PATH = r"uis_models"

##################################
# Loading the final classification model
# from the MODELS_PATH
##################################
final_cnn_model = load_model(os.path.join("..",MODELS_PATH, "cdrbnr_complex_best_model.keras"))

##################################
# Recreating the CNN model defined as
# complex CNN with dropout and batch normalization regularization
# using the Functional API structure
##################################

##################################
# Defining the input layer
##################################
fcnnmodel_input_layer = Input(shape=(227, 227, 1), name="input_layer")

##################################
# Using the layers from the Sequential model
# as functions in the Functional API
##################################
set_seed()
fcnnmodel_conv2d_layer = final_cnn_model.layers[0](fcnnmodel_input_layer) # Conv2D layer
fcnnmodel_maxpooling2d_layer = final_cnn_model.layers[1](fcnnmodel_conv2d_layer) # MaxPooling2D layer
fcnnmodel_conv2d_1_layer = final_cnn_model.layers[2](fcnnmodel_maxpooling2d_layer) # Conv2D layer
fcnnmodel_maxpooling2d_1_layer = final_cnn_model.layers[3](fcnnmodel_conv2d_1_layer) # MaxPooling2D layer
fcnnmodel_conv2d_2_layer = final_cnn_model.layers[4](fcnnmodel_maxpooling2d_1_layer) # Conv2D layer
fcnnmodel_batchnormalization_layer = final_cnn_model.layers[5](fcnnmodel_conv2d_2_layer) # Batch Normalization layer
fcnnmodel_activation_layer = final_cnn_model.layers[6](fcnnmodel_batchnormalization_layer) # Activation layer
fcnnmodel_maxpooling2d_2_layer = final_cnn_model.layers[7](fcnnmodel_activation_layer) # MaxPooling2D layer
fcnnmodel_flatten_layer = final_cnn_model.layers[8](fcnnmodel_maxpooling2d_2_layer) # Flatten layer
fcnnmodel_dense_layer = final_cnn_model.layers[9](fcnnmodel_flatten_layer) # Dense layer (128 units)
fcnnmodel_dropout_layer = final_cnn_model.layers[10](fcnnmodel_dense_layer) # Dropout layer
fcnnmodel_output_layer = final_cnn_model.layers[11](fcnnmodel_dropout_layer) # Dense layer (num_classes units)

##################################
# Creating the Functional API model
##################################
final_cnn_model_functional_api = Model(inputs=fcnnmodel_input_layer, outputs=fcnnmodel_output_layer, name="final_cnn_model_fapi")

##################################
# Compiling the Functional API model
# with the same parameters
##################################
set_seed()
final_cnn_model_functional_api.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

##################################
# Creating a gradient model for the
# gradient class activation map
# of the first convolutional layer
##################################
grad_model_first_conv2d = Model(inputs=fcnnmodel_input_layer, 
                                outputs=[fcnnmodel_conv2d_layer, fcnnmodel_output_layer], 
                                name="final_cnn_model_fapi_first_conv2d")
set_seed()
grad_model_first_conv2d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

##################################
# Creating a gradient model for the
# gradient class activation map
# of the second convolutional layer
##################################
grad_model_second_conv2d = Model(inputs=fcnnmodel_input_layer, 
                                outputs=[fcnnmodel_conv2d_1_layer, fcnnmodel_output_layer], 
                                name="final_cnn_model_fapi_second_conv2d")
set_seed()
grad_model_second_conv2d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

##################################
# Creating a gradient model for the
# gradient class activation map
# of the third convolutional layer
##################################
grad_model_third_conv2d = Model(inputs=fcnnmodel_input_layer, 
                                outputs=[fcnnmodel_conv2d_2_layer, fcnnmodel_output_layer], 
                                name="final_cnn_model_fapi_third_conv2d")
set_seed()
grad_model_third_conv2d.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall(name='recall')])

##################################
# Formulating a function to
# preprocess an individually sampled test case
# for model prediction
# in terms of size, color mode and rescaling
##################################
def preprocess_image(image_path, target_size=(227, 227), color_mode="grayscale", rescale=1./255):
    # Reading the image
    image = cv2.imread(image_path)

    # Converting to grayscale if needed
    if color_mode == "grayscale":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Expanding dimensions to simulate channel (H, W, 1)
        image = np.expand_dims(image, axis=-1)

    # Resizing the image to the target size
    image = cv2.resize(image, target_size)

    # Normalizing pixel values
    image = image * rescale

    # Adding a batch dimension (1, H, W, C)
    image = np.expand_dims(image, axis=0)

    return image

##################################
# Formulating a function to
# predict the image category
# of an individually sampled test case
##################################
def predict_image(preprocessed_sample_image):
    # Gathering the predictions of the sample image
    predictions = final_cnn_model.predict(preprocessed_sample_image)
    
    # Providing a verbose description of the image classes
    classes = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
    
    # Flattening batch dimension
    predicted_probabilities = predictions[0]  
    
    # Determining the index of highest probability
    predicted_class_index = np.argmax(predicted_probabilities)  
    
    # Obtaining the predicted class label
    predicted_class = classes[predicted_class_index] 

    # Obtaining the individual class probabilities
    notumor_probability = 100*predicted_probabilities[0]
    glioma_probability = 100*predicted_probabilities[1]
    meningioma_probability = 100*predicted_probabilities[2]
    pituitary_probability = 100*predicted_probabilities[3]

    return predicted_class, notumor_probability, glioma_probability, meningioma_probability, pituitary_probability

##################################
# Defining a function
# to formulate the gradient class activation map
# from the output of the first convolutional layer
##################################
def make_gradcam_heatmap_first_conv2d(img_array, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model_first_conv2d(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds

##################################
# Defining a function
# to formulate the gradient class activation map
# from the output of the second convolutional layer
##################################
def make_gradcam_heatmap_second_conv2d(img_array, pred_index=None):    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model_second_conv2d(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds

##################################
# Defining a function
# to formulate the gradient class activation map
# from the output of the third convolutional layer
##################################
def make_gradcam_heatmap_third_conv2d(img_array, pred_index=None):    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model_third_conv2d(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds


##################################
# Defining a function
# to load the sampled images
##################################
img_size=227
def readImage(path):
    img = load_img(path,color_mode="grayscale", target_size=(img_size,img_size))
    img = img_to_array(img)
    img = img/255.    
    return img

##################################
# Defining a function
# to colorize the generated heatmap
# and superimpose on the actual image
##################################
def gradcam_image_prediction(path):
    img = readImage(path)
    img = np.expand_dims(img,axis=0)
    heatmap_first_conv2d, preds_first_conv2d = make_gradcam_heatmap_first_conv2d(img)
    heatmap_second_conv2d, preds_second_conv2d = make_gradcam_heatmap_second_conv2d(img)
    heatmap_third_conv2d, preds_third_conv2d = make_gradcam_heatmap_third_conv2d(img)

    heatmap_first_conv2d = np.uint8(255 * heatmap_first_conv2d)
    heatmap_second_conv2d = np.uint8(255 * heatmap_second_conv2d)
    heatmap_third_conv2d = np.uint8(255 * heatmap_third_conv2d)

    img = load_img(path)
    img = img_to_array(img)
    
    jet = plt.colormaps["turbo"]
    jet_colors = jet(np.arange(256))[:, :3]
    
    def process_heatmap(heatmap, img):
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * 0.80 + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        # Resizing the superimposed image to 277x277
        superimposed_img = superimposed_img.resize((277, 277))
        return superimposed_img

    superimposed_img_first_conv2d = process_heatmap(heatmap_first_conv2d, img)
    superimposed_img_second_conv2d = process_heatmap(heatmap_second_conv2d, img)
    superimposed_img_third_conv2d = process_heatmap(heatmap_third_conv2d, img)

    return superimposed_img_first_conv2d, superimposed_img_second_conv2d, superimposed_img_third_conv2d

