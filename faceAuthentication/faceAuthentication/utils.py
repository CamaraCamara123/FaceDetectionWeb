# In this script, we will set all of the methods that you're going to be using in your API endpoints

# Import the needed dependencies
from django.db import connection
from django.db.utils import OperationalError
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from faceAuthentication.api_methods.models import User
from django.views.decorators.csrf import csrf_exempt
import json
from django.shortcuts import render, redirect
from django.http import HttpResponse

import os
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
import base64

# Create our model architecture

# Define the embedding model


def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D((2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D((2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D((2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=inp, outputs=d1, name='embedding')

# Siamese L1 Distance class


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Define the embedding model

# Create a new model for multi-class classification


def make_classification_model(num_classes):
    # Input image
    input_image = Input(shape=(100, 100, 3), name='input_image')

    embedding_model = make_embedding()

    # Get the embedding
    embedding = embedding_model(input_image)

    # Classification layer
    classifier = Dense(num_classes, activation='softmax')(embedding)

    print("Finished making the model")

    return Model(inputs=input_image, outputs=classifier, name='ClassificationNetwork')


def createFinalModel(num_classes):

    # Assuming you have N classes
    classification_model = make_classification_model(num_classes)

    # Compile the classification model
    classification_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return classification_model


# Image Data Generation and Classes

def getImageData():
    # Parameters
    batch_size = 32
    img_size = (100, 100)

    # Path to the source directory
    source_dir = 'data'

    # Create an ImageDataGenerator for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Flow training images in batches of 32 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        source_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator


def train_model(num_epochs, num_classes):

    # Train the model

    # Don't forget to set the num of classes
    classification_model = createFinalModel(num_classes)
    train_generator = getImageData()

    classification_model.fit(
        train_generator,
        epochs=num_epochs
    )

    classification_model.save('model/facedetect.h5')

    return classification_model, train_generator


def classify_image(image_path):

    # Load a new image and preprocess it
    # Load and preprocess the image

    classification_model = tf.keras.models.load_model('model/facedetect.h5',
                                                      custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

    img = image.load_img(image_path, target_size=(100, 100))

    img_array = image.img_to_array(img)
    # Expand dims to make it (1, 100, 100, 3)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize

    # Predict
    predictions = classification_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the mapping of class indices to class names
    train_generator = getImageData()
    class_indices = train_generator.class_indices

    # Invert the dictionary to get a mapping from index to class name
    index_to_class = {v: k for k, v in class_indices.items()}

    # Get the predicted class name
    predicted_class_name = index_to_class[predicted_class_index]
    print(f"Predicted class: {predicted_class_name}")


LOCK_FILE = os.path.join(settings.BASE_DIR, 'register_user.lock')


@csrf_exempt
def register_user(request):
    if request.method == 'POST':
        if os.path.exists(LOCK_FILE):
            return HttpResponse("La méthode est déjà en cours d'exécution", status=429)

        try:
            # Créer un fichier de verrouillage
            with open(LOCK_FILE, 'w') as lock:
                lock.write('locked')

            # Code pour enregistrer l'utilisateur
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            username = request.POST.get('username')
            pass_phrase = request.POST.get('pass_phrase')
            photo = request.FILES.get('photo')

            if first_name and last_name and username and pass_phrase:

                if User.objects.filter(username=username).exists():
                    return HttpResponse("Le nom d'utilisateur existe déjà", status=409)
                
                if photo:
                    data_dir = os.path.join(settings.MEDIA_ROOT)
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)

                    photo_name = f"{username}.png"
                    photo_path = os.path.join(data_dir, username, photo_name)
                    if not os.path.exists(os.path.join(data_dir, username)):
                        os.makedirs(os.path.join(data_dir, username))

                    with open(photo_path, 'wb+') as destination:
                        for chunk in photo.chunks():
                            destination.write(chunk)

                user = User.objects.create(
                    first_name=first_name,
                    last_name=last_name,
                    username=username,
                    pass_phrase=pass_phrase,
                )

                users = User.objects.all().values('id', 'first_name', 'last_name', 'username')
                users_list = list(users)
                print(f"la taille {len(users_list)}")

                model = train_model(30, len(users_list))
                print(model)
                return HttpResponse("success", status=200)
            else:
                return HttpResponse("Missing fields", status=400)

        except OperationalError as e:
            return HttpResponse(f"Database error: {str(e)}", status=500)
        except Exception as e:
            return HttpResponse(f"An error occurred: {str(e)}", status=500)
        finally:
            # Supprimer le fichier de verrouillage
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)

    return HttpResponse("Invalid request method", status=405)
