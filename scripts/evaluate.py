import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os


test_dir = "data/test"

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical'
)


model_path = 'model/best_model.h5'  
model = load_model(model_path)


score = model.evaluate(test_generator)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")


import numpy as np
sample_images, _ = next(test_generator)
predictions = model.predict(sample_images)


for i, prediction in enumerate(predictions[:5]):  
    predicted_class = np.argmax(prediction)
    print(f"Sample {i + 1} predicted class: {predicted_class}")
