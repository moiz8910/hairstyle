
# import streamlit as st
# import os
# from PIL import Image, ImageOps
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.layers import GlobalMaxPooling2D
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from sklearn.neighbors import NearestNeighbors
# from numpy.linalg import norm
# import pickle

# # Define the background CSS
# background_css = """
# <style>
#   body {
#     background-color: #001f3f;
#     color: white;
#     display: flex;
#     flex-direction: row;
#     justify-content: space-between;
#     align-items: center;
#     height: 100vh;
#   }

#   .letter {
#     font-size: 200px;
#     color: #FFD700; /* Yellow color */
#     opacity: 0.5;
#   }
# </style>
# """

# # Display the background CSS
# st.markdown(background_css, unsafe_allow_html=True)

# # Display the "S" letters on the left and right
# st.write('<div class="letter">#SyncStore</div>', unsafe_allow_html=True)

# # Load your feature_list and filenames data
# feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
# filenames = pickle.load(open('filenames.pkl', 'rb'))

# model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# model.trainable = False

# feature_extractor = tf.keras.Sequential([
#     model,
#     GlobalMaxPooling2D()
# ])

# # Styling for the tagline
# st.markdown(
#     """
#     <style>
#     body {
#         background-color: #001f3f;
#         color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# st.markdown(
#     '<div style="text-align: center; padding: 20px; background-color: #FF5733;">'
#     '<p style="font-size: 24px; font-weight: bold; color: black;">'
#     'Unlock Your Ideal Hairstyle with Our Advanced Recommending System, Then Elevate It with Expert-Recommended Hair Products!'
#     '</p>'
#     '</div>',
#     unsafe_allow_html=True
# )

# def save_uploaded_file(uploaded_file):
#     try:
#         with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
#             f.write(uploaded_file.getbuffer())
#         return True
#     except Exception as e:
#         print("Error:", e)
#         return False

# def feature_extraction(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)

#     return normalized_result

# def recommend(features, feature_list):
#     neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
#     neighbors.fit(feature_list)

#     distances, indices = neighbors.kneighbors([features])

#     return indices

# # file upload -> save
# uploaded_file = st.file_uploader("Choose an image")
# if uploaded_file is not None:
#     if save_uploaded_file(uploaded_file):
#         # display the uploaded image
#         display_image = Image.open(uploaded_file)
#         uploaded_image_size = (150, 150)  # Reduced size for the uploaded image
#         display_image = display_image.resize(uploaded_image_size)
#         st.image(display_image, use_column_width=True)
#         # feature extract
#         features = feature_extraction(os.path.join("uploads", uploaded_file.name), feature_extractor)
#         # recommendation
#         indices = recommend(features, feature_list)
#         # show recommended images
#         col1, col2, col3, col4, col5 = st.columns(5)

#         recommended_image_size = (120, 120)  # Slightly increased size for recommended images

#         border_color = (0, 0, 0)  # Black border color

#         with col1:
#             recommended_image = Image.open(filenames[indices[0][0]])
#             recommended_image = recommended_image.resize(recommended_image_size)
#             recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
#             st.image(recommended_image, use_column_width=True)
#         with col2:
#             recommended_image = Image.open(filenames[indices[0][1]])
#             recommended_image = recommended_image.resize(recommended_image_size)
#             recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
#             st.image(recommended_image, use_column_width=True)
#         with col3:
#             recommended_image = Image.open(filenames[indices[0][2]])
#             recommended_image = recommended_image.resize(recommended_image_size)
#             recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
#             st.image(recommended_image, use_column_width=True)
#         with col4:
#             recommended_image = Image.open(filenames[indices[0][3]])
#             recommended_image = recommended_image.resize(recommended_image_size)
#             recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
#             st.image(recommended_image, use_column_width=True)
#         with col5:
#             recommended_image = Image.open(filenames[indices[0][4]])
#             recommended_image = recommended_image.resize(recommended_image_size)
#             recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
#             st.image(recommended_image, use_column_width=True)
#     else:
#         st.header("Some error occurred in file upload")

# # Styling for the tagline
# st.markdown(
#     '<p style="text-align: center; padding: 20px; font-size: 24px; font-weight: bold; color: #FF5733;">'
#     'Elevate Your Style with Tailored Hair Products – Click to Shine!'
#     '</p>',
#     unsafe_allow_html=True
# )

# # Center-align the "Buy Now" button and add a cart icon
# buy_now_html = """
# <div style="text-align: center;">
#   <a href="http://localhost:3000/products?category=cosmetics%20and%20body%20care" 
#      style="text-decoration: none; color: white;">
#     <button style="font-size: 18px; padding: 10px 20px; background-color: #008CBA; color: white; border: none; border-radius: 5px;">
#       Buy Now
#       <i class="fa fa-shopping-cart" style="margin-left: 5px;"></i>
#     </button>
#   </a>
# </div>
# """

# # Render the HTML for the "Buy Now" button
# st.markdown(buy_now_html, unsafe_allow_html=True)

import streamlit as st
import os
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import pickle

# Load your feature_list and filenames data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

feature_extractor = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Define the background CSS
background_css = """
<style>
  body {
    background-color: #001f3f;
    color: white;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    height: 100vh;
  }

  .letter {
    font-size: 200px;
    color: #FFD700; /* Yellow color */
    opacity: 0.5;
  }
</style>
"""

# Display the background CSS
st.markdown(background_css, unsafe_allow_html=True)

# Display the "S" letters on the left and right
st.write('<div class="letter">#YourHairstyle</div>', unsafe_allow_html=True)

# Styling for the tagline
st.markdown(
    """
    <style>
    body {
        background-color: #001f3f;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align: center; padding: 20px; background-color: #FF5733;">'
    '<p style="font-size: 24px; font-weight: bold; color: black;">'
    'Discover Your Perfect Hairstyle with Our Advanced Recommending System'
    '</p>'
    '</div>',
    unsafe_allow_html=True
)

@st.cache(allow_output_mutation=True)
def load_model():
    return feature_extractor

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        print("Error:", e)
        return False

@st.cache
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

@st.cache
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the uploaded image
        display_image = Image.open(uploaded_file)
        uploaded_image_size = (150, 150)  # Reduced size for the uploaded image
        display_image = display_image.resize(uploaded_image_size)
        st.image(display_image, use_column_width=True)
        # feature extract
        model = load_model()
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        # recommendation
        indices = recommend(features, feature_list)
        # show recommended images
        col1, col2, col3, col4, col5 = st.columns(5)

        recommended_image_size = (120, 120)  # Slightly increased size for recommended images

        border_color = (0, 0, 0)  # Black border color

        with col1:
            recommended_image = Image.open(filenames[indices[0][0]])
            recommended_image = recommended_image.resize(recommended_image_size)
            recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
            st.image(recommended_image, use_column_width=True)
        with col2:
            recommended_image = Image.open(filenames[indices[0][1]])
            recommended_image = recommended_image.resize(recommended_image_size)
            recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
            st.image(recommended_image, use_column_width=True)
        with col3:
            recommended_image = Image.open(filenames[indices[0][2]])
            recommended_image = recommended_image.resize(recommended_image_size)
            recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
            st.image(recommended_image, use_column_width=True)
        with col4:
            recommended_image = Image.open(filenames[indices[0][3]])
            recommended_image = recommended_image.resize(recommended_image_size)
            recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
            st.image(recommended_image, use_column_width=True)
        with col5:
            recommended_image = Image.open(filenames[indices[0][4]])
            recommended_image = recommended_image.resize(recommended_image_size)
            recommended_image = ImageOps.expand(recommended_image, border=5, fill=border_color)
            st.image(recommended_image, use_column_width=True)
    else:
        st.header("Some error occurred in file upload")

# Styling for the tagline
st.markdown(
    '<p style="text-align: center; padding: 20px; font-size: 24px; font-weight: bold; color: #FF5733;">'
    'Discover Your Perfect Hairstyle Today!'
    '</p>',
    unsafe_allow_html=True
)

# Center-align the "Buy Now" button and add a cart icon
buy_now_html = """
<div style="text-align: center;">
  <a href="http://localhost:3000/products?category=cosmetics%20and%20body%20care" 
     style="text-decoration: none; color: white;">
    <button style="font-size: 18px; padding: 10px 20px; background-color: #008CBA; color: white; border: none; border-radius: 5px;">
      Buy Now
      <i class="fa fa-shopping-cart" style="margin-left: 5px;"></i>
    </button>
  </a>
</div>
"""

# Render the HTML for the "Buy Now" button
st.markdown(buy_now_html, unsafe_allow_html=True)
