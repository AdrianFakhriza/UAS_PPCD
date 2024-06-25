import cv2 as cv
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import datasets, layers, models
from tensorflow.keras.utils import image_dataset_from_directory 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
import random

#PREPROCESSING : resizing, thresholding, edge detection, normalize
def thresholding(input_folder, output_folder):
    #Samakan ukuran berdasarkan pixel
    new_size = (240, 240)

    # Buat folder baru untuk menyimpan hasil resizing jika belum ada
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop melalui semua file dalam folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter file gambar
            img_path = os.path.join(input_folder, filename)

            img = cv.imread(img_path,0)

            # Resize gambar
            img_resized = cv.resize(img, new_size, interpolation=cv.INTER_LANCZOS4)

            threshold_value = 100
            ret, img_threshold = cv.threshold(img_resized, threshold_value, 255, cv.THRESH_TRUNC)
            #ret menyimpan nilai ambang yang digunakan dalam operasi thresholding = threshold_value = 100

            # Simpan hasil thresholding ke folder output
            output_path = os.path.join(output_folder, filename)
            cv.imwrite(output_path, img_threshold)

    print("Thresholding done!")

thresholding('./data_awal/YES','./dataset/YES_thresholding')
thresholding('./data_awal/NO','./dataset/NO_thresholding')

def edge_detection(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Filter file gambar
            img_path = os.path.join(input_folder, filename)

            img = cv.imread(img_path)

            #Mengubahnya jadi tipe data floating point (CV_64F) u/ menghindari hilangnya informasi tepi yg negatif
            sobelx_f = cv.Sobel(img,cv.CV_64F,1,0,ksize=3) #Gradien dalam arah x hasil dari operasi Sobel
            sobely_f = cv.Sobel(img,cv.CV_64F,0,1,ksize=3) #Gradien dalam arah y hasil dari operasi Sobel.

            magnitude = cv.magnitude(sobelx_f,sobely_f) #menghitung magnitudo gradien dari hasil deteksi tepi dalam floating point
            #abs_sobel64f = np.absolute(magnitude) #mengonversi nilai absolut magnitudo menjadi tipe data 64-bit floating point
            #sobel_8u = np.uint8(abs_sobel64f) #mengonversi hasil tersebut ke tipe data 8-bit unsigned integer

            #img = sobel_8u

            #Normalisasi gambar
            img_normalized = cv.normalize(magnitude, None, 0, 1, cv.NORM_MINMAX)

            # Simpan hasil thresholding ke folder output
            output_path = os.path.join(output_folder, filename)
            cv.imwrite(output_path, (img_normalized * 255).astype(np.uint8))

    print("Edge detection and Normalization done!")

#edge_detection('./dataset/YES_thresholding','./dataset/YES_edge_detection')
#edge_detection('./dataset/NO_thresholding','./dataset/NO_edge_detection')

edge_detection('./dataset/YES_thresholding','./dataset/data/ya')
edge_detection('./dataset/NO_thresholding','./dataset/data/tidak')

data = './dataset/data'
kelas = os.listdir(data)
print(kelas)

#Menampilkan beberapa citra beserta labelnya
yes = 'dataset/data/ya'
no = 'dataset/data/tidak'

#mengambil images dari yes dan no untuk diberi label 1 untuk yes, 0 untuk 0, dan diletakkan ke dalam list all_images
all_images = []

for filename in os.listdir(yes):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(yes, filename)
        all_images.append((img_path, 1))

for filename in os.listdir(no):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(no, filename)
        all_images.append((img_path, 0))

random.shuffle(all_images)

plt.figure(figsize=(10, 10))
for i in range(min(25, len(all_images))):
    img_path, label = all_images[i]
    img = cv.imread(img_path)
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.xlabel('Ada Ikan' if label == 1 else 'Tidak Ada Ikan')

plt.tight_layout()
plt.show()

#Membagi data menjadi data train dan data test
train_datagen = image_dataset_from_directory(data,
                                             image_size=(240,240),
                                             subset='training',
                                             seed = 1, #untuk konsistensi gambar mana saja yang jadi train dan test (akan selalu sama)
                                             validation_split = 0.2,
                                             batch_size = 32)

test_datagen = image_dataset_from_directory(data,
                                            image_size=(240,240),
                                            subset='validation',
                                            seed = 1, #untuk konsistensi gambar mana saja yang jadi train dan test (akan selalu sama)
                                            validation_split = 0.2,
                                            batch_size = 32)

#Membuat Lapisan Konvolusional diikuti MaxPooling untuk mengekstraksi fitur dari citra 
#layer Flatten u/ mengubah  output dari layer konvolusi menjadi vektor satu dimensi
#layer Dense di bagian akhir untuk klasifikasi, dengan fungsi aktivasi ReLU untuk layer tersembunyi dan sigmoid untuk output

img_height, img_width = 240, 240
new_size = (img_width, img_height)

# Definisikan model
model = models.Sequential()
#layer konvolusi pertama dengan 32 jumlah filter, kernel 3x3, fungsi aktivasi 'ReLU setelah operasi konvolusi. Input model 240, 240, 3(gambar channel 3/berwarna)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2))) #ukuran jendela pooling 2x2
#layer konvolusi tambahan menggunakan 64 filter dan fungsi aktivitas ReLU
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#mengubah output layer sebelumnya menjadi vektor satu dimensi agar dapat jadi masukan pada layer Dense (fully connected)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#layer output dengan 1 unit (karena klasifikasi biner)
#fungsi sigmoid menghasilkan probabilitas 0 hingga 1
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile( 
    loss='binary_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'] 
)

# Callback untuk menyimpan model terbaik berdasarkan akurasi validasi
checkpoint_callback = ModelCheckpoint(filepath='best_model.keras',
                                      monitor='val_accuracy',
                                      mode='max',
                                      save_best_only=True,
                                      verbose=1)

#Pelatihan model
history = model.fit(train_datagen, 
                    epochs=15, 
                    validation_data=test_datagen,
                    callbacks=[checkpoint_callback])

test_loss, test_acc = model.evaluate(test_datagen, verbose=2)
print('Validation loss : ', test_loss)
print('Validation accuracy : ', test_acc)

history_df = pd.DataFrame(history.history) 

# Subplot 2: Plot accuracy dan val_accuracy
plt.subplot(2, 1, 1)
history_df[['accuracy', 'val_accuracy']].plot(ax=plt.gca())
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy vs Validation Accuracy')

# Subplot 2 : Plot loss dan val_loss
plt.subplot(2, 1, 2)
history_df[['loss', 'val_loss']].plot(ax=plt.gca())
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss')

# Menambahkan judul utama
plt.suptitle('Accuracy and Loss')
plt.tight_layout()
plt.show()

# Load the best model
best_model = keras.models.load_model('best_model.keras')

# Fungsi preprocessing untuk prediksi
def preprocess_image(input_folder, filename):
    img_path = os.path.join(input_folder, filename)
    
    img = cv.imread(img_path, 0)
    
    #resize image ke piksel 240 x 240
    img_height, img_width = 240, 240
    new_size = (img_width, img_height)
    img_resized = cv.resize(img, new_size, interpolation=cv.INTER_LANCZOS4)

    # Thresholding Trunc
    threshold_value = 100
    ret, img_threshold = cv.threshold(img_resized, threshold_value, 255, cv.THRESH_TRUNC)

    # Sobel u/ mendapatkan gradien dalam arah horizontal dan vertikal untuk mendapatkan gradien dalam arah horizontal dan vertikal
    # Sobel magnitude untuk edge detection
    sobelx_f = cv.Sobel(img_threshold, cv.CV_64F, 1, 0, ksize=3)
    sobely_f = cv.Sobel(img_threshold, cv.CV_64F, 0, 1, ksize=3)
    magnitude = cv.magnitude(sobelx_f, sobely_f)

    # Normalize magnitude jadi rentang [0, 1] atau min = 0 dan max = 1 dengan NORMALISASI MIN MAX dari openCV
    img_normalized = cv.normalize(magnitude, None, 0, 1, cv.NORM_MINMAX)

    # mengubahnya jadi RGB karena dia 3-channel sebab model butuh inputan 3-channel
    img_normalized_rgb = cv.cvtColor((img_normalized * 255).astype(np.uint8), cv.COLOR_GRAY2RGB)

    return img_normalized_rgb

def predict_and_plot(files):
    plt.figure(figsize=(20, 20))
    for i, file in enumerate(files):
        input_folder, filename = os.path.split(file) #di-split agar mendapatkan jalur file sehingga didapatkan direktor dan filname
        
        # Tahapan preprocessing (resizing, thresholding, edge detection, normalization)
        img_normalized = preprocess_image(input_folder, filename)
        
        #Memprediksi citra dengan model
        img_for_prediction = np.expand_dims(img_normalized, axis=0)
        predictions = model.predict(img_for_prediction)
        label = 'Ikan' if predictions[0][0] > 0.5 else 'Tidak ada ikan' #jika hasil prediksi >0.5, gambar diprediksi 'Ikan' sebaliknya, 'TIdak Ada Ikan'
        
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_normalized)
        plt.xlabel(label)
    
    plt.show()

#Fungsi mengambil gambar acak dari data awal untuk mengecek hasil prediksi oleh model
def get_random_files(dataset_folder, num_files=25):
    all_files = []
    for label in ['YES', 'NO']:
        folder_path = os.path.join(dataset_folder, label)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        all_files.extend(files)
    
    random_files = random.sample(all_files, num_files)
    return random_files


dataset_folder = './data_awal'  
random_files = get_random_files(dataset_folder)
predict_and_plot(random_files)
