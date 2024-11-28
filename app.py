import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the MobileNetV2 model for feature extraction
mobilenet_model = MobileNetV2(weights="imagenet")
mobilenet_model = tf.keras.Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

# Load your trained model
model = tf.keras.models.load_model('best_model.h5')

# Load the tokenizer
with open('features.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load and preprocess the image
image_path = 'image.jpg' 
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# Extract features from the image
image_features = mobilenet_model.predict(image, verbose=0)

# Define function to get word from index
def get_word_from_index(index, tokenizer):
    return next((word for word, idx in tokenizer.word_index.items() if idx == index), None)

# Generate caption using the model
def predict_caption(model, image_features, tokenizer, max_caption_length):
    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)
        if predicted_word is None:
            break
        caption += " " + predicted_word
        if predicted_word == "endseq":
            break
    return caption

# Max caption length
max_caption_length = 34

# Generate and print the caption
generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

# Clean up the caption
generated_caption = generated_caption.replace("startseq", "").replace("endseq", "").strip()

print("Generated Caption:", generated_caption)
