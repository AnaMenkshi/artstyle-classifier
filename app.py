import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os

print("Files in the current directory:", os.listdir('/home/user/app/'))

# Load the trained model
model = tf.keras.models.load_model('/home/user/app/Art_model.keras')

# Class labels corresponding to the model's output indices
class_labels = ['Romanticism', 'Realism', 'Post-Impressionism', 'Expressionism', 'Impressionism']

# Preprocess the image to match model input requirements
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to the expected input size of the model
    image = np.array(image) / 255.0  # Normalize image if required by your model
    image = np.expand_dims(image, axis=0)  # Add batch dimension for prediction
    return image

# Define the prediction function
def predict(image):
    processed_image = preprocess_image(image)  # Preprocess the uploaded image
    predictions = model.predict(processed_image)  # Get model predictions
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest probability
    predicted_class = class_labels[predicted_class_index]  # Map the index to the corresponding style label
    confidence = predictions[0][predicted_class_index]  # Confidence score for the predicted class
    return predicted_class, f"{confidence:.2%}"  # Return the style and formatted confidence

# Authentication function
def authenticate(username, password):
    valid_username = "art"
    valid_password = "art"
    
    # Check if the provided credentials are correct
    return username == valid_username and password == valid_password

# Create the Gradio interface with authentication
with gr.Blocks() as demo:
    # Authentication status
    auth_status = gr.Textbox(label="Authentication Status", interactive=False, visible=False)
    
    # Authentication section
    with gr.Row():
        username_input = gr.Textbox(label="Username", placeholder="Enter your username")
        password_input = gr.Textbox(label="Password", placeholder="Enter your password", type="password")
        auth_button = gr.Button("Login")
    
    # Prediction section (hidden initially)
    with gr.Row(visible=False) as prediction_section:
        input_image = gr.Image(type="pil", label="Upload an image")
        predicted_label = gr.Textbox(label="Predicted Style")
        confidence_score = gr.Textbox(label="Confidence Score")
        input_image.change(predict, inputs=input_image, outputs=[predicted_label, confidence_score])
    
    # Button actions
    def handle_auth(username, password):
        if authenticate(username, password):
            return "Authentication Successful!", gr.update(visible=True)  # Show prediction section
        else:
            return "Authentication Failed. Try Again.", gr.update(visible=False)  # Hide prediction section
    
    auth_button.click(
        handle_auth, 
        inputs=[username_input, password_input], 
        outputs=[auth_status, prediction_section]
    )

# Launch the app
demo.launch()
