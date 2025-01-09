##################################
# Loading Python Libraries
##################################
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import cv2
from glob import glob
import time
from pathlib import Path

from model_prediction_community_cloud_app import preprocess_image, predict_image
from model_prediction_community_cloud_app import make_gradcam_heatmap_first_conv2d, make_gradcam_heatmap_second_conv2d, make_gradcam_heatmap_third_conv2d
from model_prediction_community_cloud_app import gradcam_image_prediction

##################################
# Defining file paths
##################################
DATASETS_FINAL_TEST_PATH = Path("uis_datasets/Brain_Tumor_MRI_Dataset/Testing")

##################################
# Defining the image category levels
# for the testing data
##################################
diagnosis_code_dictionary_test = {'Te-no': 0,
                                  'Te-noTr': 0,
                                  'Te-gl': 1,
                                  'Te-glTr': 1,
                                  'Te-me': 2,
                                  'Te-meTr': 2,
                                  'Te-pi': 3,
                                  'Te-piTr': 3}

##################################
# Defining the image category descriptions
# for the testing data
##################################
diagnosis_description_dictionary_test = {'Te-no': 'No Tumor',
                                         'Te-noTr': 'No Tumor',
                                         'Te-gl': 'Glioma',
                                         'Te-glTr': 'Glioma',
                                         'Te-me': 'Meningioma',
                                         'Te-meTr': 'Meningioma',
                                         'Te-pi': 'Pituitary',
                                         'Te-piTr': 'Pituitary'}

##################################
# Consolidating the image path
# for the testing data
##################################
imageid_path_dictionary_test = {
    path.stem: path for path in DATASETS_FINAL_TEST_PATH.glob("*/**/*.jpg")
}

##################################
# Consolidating the information
# from the testing data
# into a dataframe
##################################
mri_images_test = pd.DataFrame.from_dict(imageid_path_dictionary_test, orient = 'index').reset_index()
mri_images_test.columns = ['Image_ID','Path']
classes = mri_images_test.Image_ID.str.split('_').str[0]
mri_images_test['Diagnosis'] = classes
mri_images_test['Target'] = mri_images_test['Diagnosis'].map(diagnosis_code_dictionary_test.get) 
mri_images_test['Class'] = mri_images_test['Diagnosis'].map(diagnosis_description_dictionary_test.get)

##################################
# Setting the page layout to wide
##################################
st.set_page_config(layout="wide")

##################################
# Initializing session state 
# for button and variable tracking
##################################
if "sample_clicked" not in st.session_state:
    st.session_state.sample_clicked = False
if "picture" not in st.session_state:
    st.session_state.picture = None
if "resized_image" not in st.session_state:
    st.session_state.resized_image = None
if "original_class" not in st.session_state:
    st.session_state.original_class = None

##################################
# Creating a title for the application
##################################
st.markdown("""---""")
st.markdown("<h1 style='text-align: center;'>Brain Magnetic Resonance Image Classifier</h1>", unsafe_allow_html=True)

##################################
# Providing a description for the application
##################################
st.markdown("""---""")
st.markdown("<h5 style='font-size: 20px;'>This model leverages the convolutional neural network architecture to classify a magnetic resonance image by directly learning hierarchical features from raw pixel data and extracting low- and high-level features for differentiating between image categories. Randomly sample an image from a test cohort to determine the predicted class, estimate all individual class probabilities and generate an advanced visualization using Gradient Class Activation Mapping (Grad-CAM) to aid in providing insights into the spatial and hierarchical features that influenced the model's predictions and offering a deeper understanding of the decision-making process. For more information on the complete model development process, you may refer to this <a href='https://johnpaulinepineda.github.io/Portfolio_Project_56/' style='font-weight: bold;'>Jupyter Notebook</a>. Additionally, all associated datasets and code files can be accessed from this <a href='https://github.com/JohnPaulinePineda/Portfolio_Project_56' style='font-weight: bold;'>GitHub Project Repository</a>.</h5>", unsafe_allow_html=True)

##################################
# Defining the action buttons
##################################  
st.markdown("""---""")

st.markdown("""
    <style>
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

##################################
# Setting the sample button
# as initially enabled
################################## 
if st.button("Generate a Brain Magnetic Resonance Image Sample"):
    ##################################
    # Activating the predict button
    ##################################
    st.session_state.sample_clicked = True  
    
    ##################################
    # Defining the code logic
    # for the sample button action
    ################################## 

    ##################################
    # Setting a dynamic seed
    ##################################
    random.seed(time.time())
    
    ##################################
    # Sampling a single image
    # from the testing data
    ##################################
    samples, features = mri_images_test.shape
    pic_id = random.randrange(0, samples)
    picture = mri_images_test['Path'][pic_id]
    image = cv2.imread(picture)
    original_class = mri_images_test['Class'][pic_id]

    ##################################
    # Defining a section title
    # for the test image channel visualization
    ##################################      
    st.markdown("""---""")   
    st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Image RGB Channel Visualization</h4>", unsafe_allow_html=True)    
    st.markdown("""---""")

    ##################################
    # Plotting the sampled image
    # from the testing data
    ##################################

    ##################################
    # Plotting using subplots
    ##################################
    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    ##################################
    # Resizing the image
    ##################################
    resized_image = cv2.resize(image, (227, 227))

    ##################################
    # Storing the variables in session_state
    ##################################
    st.session_state.picture = picture
    st.session_state.resized_image = resized_image
    st.session_state.original_class = original_class

    ##################################
    # Formulating the original image
    ##################################
    axs[0].imshow(resized_image)
    axs[0].set_title('Original Image', fontsize=10, weight='bold')
    axs[0].axis('off')

    ##################################
    # Formulating the red channel
    ##################################
    red_channel = np.zeros_like(resized_image)
    red_channel[:, :, 0] = resized_image[:, :, 0]
    axs[1].imshow(red_channel)
    axs[1].set_title('Red Channel', fontsize=10, weight='bold')
    axs[1].axis('off')

    ##################################
    # Formulating the green channel
    ##################################
    green_channel = np.zeros_like(resized_image)
    green_channel[:, :, 1] = resized_image[:, :, 1]
    axs[2].imshow(green_channel)
    axs[2].set_title('Green Channel', fontsize=10, weight='bold')
    axs[2].axis('off')

    ##################################
    # Formulating the blue channel
    ##################################
    blue_channel = np.zeros_like(resized_image)
    blue_channel[:, :, 2] = resized_image[:, :, 2]
    axs[3].imshow(blue_channel)
    axs[3].set_title('Blue Channel', fontsize=10, weight = 'bold')
    axs[3].axis('off')

    ##################################
    # Consolidating all images
    ##################################
    st.pyplot(fig)


st.markdown("""---""")


##################################
# Setting the predict button
# as initially disabled
##################################
if not st.session_state.sample_clicked:
    st.button("Determine Predicted Class + Estimate Class Probabilities + Visualize Gradient Class Activation Mapping", disabled=True)
else:
    ##################################
    # Defining the code logic
    # for the sample button action
    ################################## 
    if st.button("Determine Predicted Class + Estimate Class Probabilities + Visualize Gradient Class Activation Mapping"):
        
        ##################################
        # Using stored variables
        ##################################
        picture = st.session_state.picture
        resized_image = st.session_state.resized_image
        original_class = st.session_state.original_class

        ##################################
        # Defining a section title
        # for the test image Grad-CAM visualization
        ##################################      
        st.markdown("""---""")   
        st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Image Gradient Class Activation Mapping by Convolutional Layer</h4>", unsafe_allow_html=True)    
        st.markdown("""---""")
        
        ##################################
        # Plotting using subplots
        ##################################
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))

        ##################################
        # Preprocessing the sampled image
        ##################################
        preprocessed_image = preprocess_image(picture, target_size=(227, 227), color_mode="grayscale")

        ##################################
        # Obtaining the class prediction
        ##################################
        predicted_class, notumor_probability, glioma_probability, meningioma_probability, pituitary_probability = predict_image(preprocessed_image)

        ##################################
        # Obtaining the GradCAM of the sampled image
        ##################################
        grad_image_first_conv2d, grad_image_second_conv2d, grad_image_third_conv2d = gradcam_image_prediction(picture)

        ##################################
        # Formulating the original image
        ##################################
        axs[0].imshow(resized_image)
        axs[0].set_title('Original Image', fontsize=10, weight='bold')
        axs[0].axis('off')

        ##################################
        # Formulating the red channel
        ##################################
        axs[1].imshow(grad_image_first_conv2d)
        axs[1].set_title('First Conv2D', fontsize=10, weight='bold')
        axs[1].axis('off')

        ##################################
        # Formulating the green channel
        ##################################
        axs[2].imshow(grad_image_second_conv2d)
        axs[2].set_title('Second Conv2D', fontsize=10, weight='bold')
        axs[2].axis('off')

        ##################################
        # Formulating the blue channel
        ##################################
        axs[3].imshow(grad_image_third_conv2d)
        axs[3].set_title('Third Conv2D', fontsize=10, weight = 'bold')
        axs[3].axis('off')

        ##################################
        # Consolidating all images
        ##################################
        st.pyplot(fig)

        ##################################
        # Defining a section title
        # for the test case model prediction summary
        ##################################      
        st.markdown("""---""")   
        st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>Test Image Model Prediction Summary</h4>", unsafe_allow_html=True)    
        st.markdown("""---""")

        ##################################
        # Summarizing the test case model prediction results
        ##################################     
        if original_class == predicted_class:
            st.markdown(f"<h4 style='font-size: 20px;'> <span style='color:blue;'>Predicted class is correct.</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Original Class: <span style='color:blue;'>{original_class}</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Predicted Class: <span style='color:blue;'>{predicted_class}</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>No Tumor Class Probability: <span style='color:blue;'>{notumor_probability:.5f}%</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Glioma Class Probability: <span style='color:blue;'>{glioma_probability:.5f}%</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Meningioma Class Probability: <span style='color:blue;'>{meningioma_probability:.5f}%</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Pituitary Class Probability: <span style='color:blue;'>{pituitary_probability:.5f}%</span></h4>", unsafe_allow_html=True)
    
        if original_class != predicted_class:
            st.markdown(f"<h4 style='font-size: 20px;'> <span style='color:red;'>Predicted class is incorrect.</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Original Class: <span style='color:red;'>{original_class}</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Predicted Class: <span style='color:red;'>{predicted_class}</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>No Tumor Class Probability: <span style='color:red;'>{notumor_probability:.5f}%</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Glioma Class Probability: <span style='color:red;'>{glioma_probability:.5f}%</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Meningioma Class Probability: <span style='color:red;'>{meningioma_probability:.5f}%</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='font-size: 20px;'>Pituitary Class Probability: <span style='color:red;'>{pituitary_probability:.5f}%</span></h4>", unsafe_allow_html=True)

        
st.markdown("""---""")
        


