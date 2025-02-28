import streamlit as st
import tensorflow as tf
import numpy as np

# Custom CSS for Green & Dark Theme
st.markdown(
    """
    <style>
        body {
            background-color: #1B2A1F;
            color: #DFFFD8;
        }
        .sidebar .sidebar-content {
            background-color: #2C3E2D;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #B0F4B0;
        }
        .stButton>button {
            background-color: #3D9970;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #2E7D5F;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("🌿 Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Page", ["Home", "About", "Disease Detection"], index=0)

# Main Page
if app_mode == "Home":
    st.title("🌱 Plant Disease Detection System")
    image_path = "public/home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown(
        """
        Welcome to the **Plant Disease Detection System**! 🍃🔍
        
        Our system helps identify plant diseases efficiently. Upload an image, and let our AI analyze it!
        
        **🌟 How It Works:**
        1. Upload an image on the **Disease Detection** page.
        2. Our AI processes the image to detect diseases.
        3. Get instant results and take necessary action.
        
        **🌿 Why Choose Us?**
        ✅ High Accuracy
        ✅ Fast Analysis
        ✅ Easy to Use
        """
    )

# About Page
elif app_mode == "About":
    st.title("📌 About the Project")
    st.markdown(
        """
        ### Dataset Information
        - 87K RGB images of healthy & diseased leaves.
        - Categorized into **38 different classes**.
        - Dataset split: **80% Training, 20% Validation**.
        """
    )

# Prediction Page
elif app_mode == "Disease Detection":
    st.title("🔍 Disease Detection")
    test_image = st.file_uploader("📷 Choose a Plant Image:")
    if test_image:
        st.image(test_image, width=300)
    
    if st.button("🟢 Predict Disease"):
        st.snow()
        st.write("🧠 Analyzing... Please wait.")
        result_index = model_prediction(test_image)
        
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']
        st.success(f"🌿 Disease Identified: **{class_name[result_index]}**")
    