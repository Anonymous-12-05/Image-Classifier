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
    st.write("This project is an image classification application using the CIFAR-10 dataset and a convolutional neural network (CNN) model.")
    
    st.header("Objective")
    st.write("The objective of this project is to classify images into one of the ten categories present in the CIFAR-10 dataset.")

    st.header("Dataset")
    st.write("The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 testing images.")

    st.header("Model")
    st.write("The convolutional neural network (CNN) model used in this project consists of two convolutional layers with max-pooling followed by two fully connected dense layers. The model architecture is as follows:")
  

    st.header("Execution")
    st.write("To use the application, upload an image using the provided file uploader. Once the image is uploaded, click the 'Predict' button to classify the image.")

    st.header("Technology Stack")
    st.write("The project is built using Python and the following libraries:")
    st.write("- TensorFlow: For building and training the CNN model.")
    st.write("- Streamlit: For creating the interactive web application.")
    st.write("- NumPy: For numerical computations.")
    st.write("- Matplotlib: For data visualization.")
    st.write("- PIL (Python Imaging Library): For image processing.")

    st.header("Conclusion")
    st.write("This project demonstrates the use of deep learning techniques for image classification tasks. It showcases the integration of a trained CNN model into a Streamlit web application, providing a user-friendly interface for image classification.")

    st.header("References")
    st.write("1. CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html")
    st.write("2. TensorFlow documentation: https://www.tensorflow.org/")
    st.write("3. Streamlit documentation: https://docs.streamlit.io/")


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
