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
    st.write("The convolutional neural network (CNN) model used in this project consists of two convolutional layers with max-pooling followed by two fully connected dense layers.")
  

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
    st.write("The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.")
    
    st.header("Classes")
    st.write("The dataset contains the following classes:")
    st.write(classes)
    
    st.header("Description")
    st.write("CIFAR-10 is a widely used dataset for image classification tasks. It contains images across 10 classes, each representing a specific object or animal.")

def model_page():
    st.title("Model")
    st.write("The convolutional neural network (CNN) model used in this project is designed for image classification tasks on the CIFAR-10 dataset.")

    st.header("Convolutional Layers")
    st.write("Convolutional layers are the core building blocks of CNNs, responsible for feature extraction.")
    st.write("- **Convolutional Layer 1:**")
    st.write("  - Filters: 32")
    st.write("  - Kernel Size: 3x3")
    st.write("  - Activation Function: ReLU")
    st.write("  - Input Shape: 32x32x3 (RGB images)")
    st.write("  - Explanation: This layer applies a set of 32 filters to the input image, each filter detecting specific features.")
    st.write("- **Max-Pooling Layer 1:**")
    st.write("  - Pool Size: 2x2")
    st.write("  - Purpose: Reduces spatial dimensions, retains important features.")
    st.write("  - Explanation: This layer reduces the spatial dimensions of the feature maps while retaining important features.")
    st.write("- **Convolutional Layer 2:**")
    st.write("  - Filters: 64")
    st.write("  - Kernel Size: 3x3")
    st.write("  - Activation Function: ReLU")
    st.write("  - Explanation: This layer further extracts features using 64 filters applied to the feature maps from the previous layer.")
    st.write("- **Max-Pooling Layer 2:**")
    st.write("  - Pool Size: 2x2")
    st.write("  - Explanation: Similar to the first max-pooling layer, this layer further reduces spatial dimensions.")

    st.header("Fully Connected Dense Layers")
    st.write("Fully connected dense layers are responsible for classification based on the extracted features.")
    st.write("- **Flatten Layer:**")
    st.write("  - Purpose: Flattens the output from the previous layer into a 1D array.")
    st.write("  - Explanation: This layer converts the multi-dimensional feature maps into a one-dimensional array for input to the dense layers.")
    st.write("- **Dense Layer 1:**")
    st.write("  - Units: 64")
    st.write("  - Activation Function: ReLU")
    st.write("  - Explanation: This layer consists of 64 neurons with a ReLU activation function for feature transformation.")
    st.write("- **Dense Layer 2:**")
    st.write("  - Units: 10 (Number of classes in CIFAR-10)")
    st.write("  - Activation Function: Softmax")
    st.write("  - Explanation: The final dense layer consists of 10 neurons, one for each class in CIFAR-10, with a softmax activation function for probability distribution over classes.")

    st.header("Optimizer")
    st.write("The optimizer is responsible for updating the weights of the neural network during training to minimize the loss function.")
    st.write("- **Optimizer:** Adam")
    st.write("  - Explanation: Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm that combines the advantages of AdaGrad and RMSProp.")

    st.header("Loss Function")
    st.write("The loss function computes the error between the predicted and actual labels during training.")
    st.write("- **Loss Function:** Sparse Categorical Crossentropy")
    st.write("  - Explanation: Sparse Categorical Crossentropy is used for multi-class classification tasks with integer labels.")

    st.header("Metrics")
    st.write("Metrics are used to evaluate the performance of the model.")
    st.write("- **Metrics:** Accuracy")
    st.write("  - Explanation: Accuracy measures the proportion of correctly classified images out of the total number of images.")

    st.header("Conclusion")
    st.write("This section provides a detailed explanation of the model architecture, optimizer, loss function, and metrics used in the CNN model for image classification.")


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
