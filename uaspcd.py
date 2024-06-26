import streamlit as st
import os
import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.title('Underwater Image Processing and Prediction')
st.set_option('deprecation.showPyplotGlobalUse', False)

input_folder_yes = st.sidebar.text_input('Input folder for YES images', 'dataset/data/YES')
input_folder_no = st.sidebar.text_input('Input folder for NO images', 'dataset/data/NO')
output_folder_thresholding_yes = 'dataset/data/YES_thresholding'
output_folder_thresholding_no = 'dataset/data/NO_thresholding'
output_folder_edge_detection_yes = 'dataset/data/YES_edge_detection'
output_folder_edge_detection_no = 'dataset/data/NO_edge_detection'
output_folder_normalized_yes = 'dataset/data/ya'
output_folder_normalized_no = 'dataset/data/tidak'

# Fungsi untuk thresholding gambar
def thresholding(input_folder, output_folder):
    new_size = (640, 480)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = cv.imread(img_path, 0)
            img_resized = cv.resize(img, new_size, interpolation=cv.INTER_LANCZOS4)
            threshold_value = 127
            ret, img_threshold = cv.threshold(img_resized, threshold_value, 255, cv.THRESH_TRUNC)
            output_path = os.path.join(output_folder, filename)
            cv.imwrite(output_path, img_threshold)

    st.write("Thresholding done for folder:", input_folder)

# Fungsi untuk deteksi tepi
def edge_detection(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = cv.imread(img_path)
            sobelx_f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
            sobely_f = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
            magnitude = cv.magnitude(sobelx_f, sobely_f)
            abs_sobel64f = np.absolute(magnitude)
            sobel_8u = np.uint8(abs_sobel64f)
            img = sobel_8u
            output_path = os.path.join(output_folder, filename)
            cv.imwrite(output_path, img)

    st.write("Edge detection done for folder:", input_folder)

# Fungsi untuk normalisasi gambar
def normalized(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            img = cv.imread(img_path)
            sobelx_f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
            sobely_f = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
            magnitude = cv.magnitude(sobelx_f, sobely_f)
            img_normalized = cv.normalize(magnitude, None, 0, 1, cv.NORM_MINMAX)
            output_path = os.path.join(output_folder, filename)
            cv.imwrite(output_path, (img_normalized * 255).astype(np.uint8))

    st.write("Normalize done for folder:", input_folder)

# Apply preprocessing steps
if st.sidebar.button('Apply'):
    thresholding(input_folder_yes, output_folder_thresholding_yes)
    thresholding(input_folder_no, output_folder_thresholding_no)
    edge_detection(output_folder_thresholding_yes, output_folder_edge_detection_yes)
    edge_detection(output_folder_thresholding_no, output_folder_edge_detection_no)
    normalized(output_folder_edge_detection_yes, output_folder_normalized_yes)
    normalized(output_folder_edge_detection_no, output_folder_normalized_no)

    # Load dan proses dataset
    yes = output_folder_normalized_yes
    no = output_folder_normalized_no

    all_images = []

    for filename in os.listdir(yes):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(yes, filename)
            all_images.append((img_path, 1))

    for filename in os.listdir(no):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(no, filename)
            all_images.append((img_path, 0))

    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

    img_width, img_height = 320, 240
    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        'dataset/data',
        classes=['ya', 'tidak'],
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        'dataset/data',
        classes=['ya', 'tidak'],
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        subset='validation'
    )

    class_names = ['YES', 'NO']

    

    # Menampilkan contoh gambar dari dataset
    train_images, train_labels = next(train_generator)
    val_images, val_labels = next(val_generator)

    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(train_images))):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        plt.xlabel(class_names[int(train_labels[i])])
    st.pyplot()

    st.write('Sedang melatih model . . .')
    
    # Definisikan model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    # Melatih model
    history = model.fit(train_generator, epochs=10, 
                        validation_data=val_generator)

    # Plot hasil training
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    
    st.pyplot()

    # Evaluasi model
    test_loss, test_acc = model.evaluate(val_generator, verbose=2)
    st.write("Akurasi : ", test_acc)

    # Fungsi preprocessing untuk prediksi
    def preprocess_image(image_path):
        img = cv.imread(image_path, 0)
        img_height, img_width = 320, 240
        new_size = (img_width, img_height)

        # Resize gambar
        img_resized = cv.resize(img, new_size, interpolation=cv.INTER_LANCZOS4)

        threshold_value = 127
        ret, img_threshold = cv.threshold(img_resized, threshold_value, 255, cv.THRESH_TRUNC)

        sobelx_f = cv.Sobel(img_threshold, cv.CV_64F, 1, 0, ksize=3)
        sobely_f = cv.Sobel(img_threshold, cv.CV_64F, 0, 1, ksize=3)
        magnitude = cv.magnitude(sobelx_f, sobely_f)

        # Normalize magnitude to [0, 1]
        img_normalized = cv.normalize(magnitude, None, 0, 1, cv.NORM_MINMAX)
        img_normalized = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
        img_normalized = np.stack((img_normalized,)*3, axis=-1)  # Repeat grayscale image to match input shape
        img_normalized = np.expand_dims(img_normalized, axis=0)   # Add batch dimension

        return img_normalized

    img_width, img_height = 320, 240

    def preprocess_image(image_path):
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  
        img = cv.resize(img, (img_width, img_height), interpolation=cv.INTER_LANCZOS4)
        img = img.astype(np.float32) / 255.0  
        img = np.expand_dims(img, axis=0)
        return img


    def predict_and_plot(files):
        plt.figure(figsize=(20, 20))
        for i, file in enumerate(files):
            img_normalized = preprocess_image(file)
            predictions = model.predict(img_normalized)
            label = 'Ikan' if predictions[0][0] < 0.2 else 'Tidak ada ikan'  

            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            img = cv.imread(file)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.xlabel(label)
        st.write('Result :')
        st.pyplot()

    def get_random_files(dataset_folder, num_files=25):
        all_files = []
        for label in ['ya', 'tidak']:
            folder_path = os.path.join(dataset_folder, label)
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            all_files.extend(files)
        
        random_files = random.sample(all_files, num_files)
        return random_files


    dataset_folder = 'dataset/data'  
    random_files = get_random_files(dataset_folder)

    predict_and_plot(random_files)
