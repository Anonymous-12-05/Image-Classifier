import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model("cifar10_cnn_model.h5")

# Define CIFAR-10 class names
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def predict_image(image):
    # Preprocess the image
    image = np.array(image.resize((32, 32))) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the class
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    return classes[predicted_class], confidence

def overview_page():
    st.title("Overview")
    st.write("Brief overview of the project.")

def dataset_page():
    st.title("Dataset")
    st.write("Description of the dataset used.")

def model_page():
    st.title("Model")
    st.write("Description of the model architecture and layers.")

def execution_page():
    st.title("Execution")
    st.write("Upload an image and predict its class.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            class_name, confidence = predict_image(image)
            st.success(f"Prediction: {class_name}, Confidence: {confidence:.2f}")

def main():
    st.sidebar.title("Navigation")
    pages = {
        "Overview": overview_page,
        "Dataset": dataset_page,
        "Model": model_page,
        "Execution": execution_page
    }

    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Execute the selected page function
    pages[selected_page]()

if __name__ == "__main__":
    main()
