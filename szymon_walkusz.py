# %% Pobieranie danych
# 1
import os, shutil
# Katalog, w którym umieszczony jest mniejszy zbiór danych
base_dir = 'C:/Users/szymo/Downloads/projekt/projekt/Nowy folder/cats_and_dogs_small'

# Katalogi podzbiorów (zbioru treningowego, walidacyjnego i testowego).
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Katalog z treningowym zbiorem zdjęć kotów.
train_cats_dir = os.path.join(train_dir, 'cats')
# Katalog z treningowym zbiorem zdjęć psów.
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Katalog z walidacyjnym zbiorem zdjęć kotów.
validation_cats_dir = os.path.join(validation_dir, 'cats')
# Katalog z walidacyjnym zbiorem zdjęć psów.
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Katalog z testowym zbiorem zdjęć kotów.
test_cats_dir = os.path.join(test_dir, 'cats')
# Katalog z testowym zbiorem zdjęć psów.
test_dogs_dir = os.path.join(test_dir, 'dogs')
 
# %% Sprawdzenie liczby zdjęć w poszczególnych folderach
# 2
print('liczba obrazów treningowych kotów:', len(os.listdir(train_cats_dir)))
print('liczba obrazów treningowych psów:', len(os.listdir(train_dogs_dir)))
print('liczba obrazów walidacyjnych kotów:', len(os.listdir(validation_cats_dir)))
print('liczba obrazów walidacyjnych psów:', len(os.listdir(validation_dogs_dir)))
print('liczba obrazów testowych kotów:', len(os.listdir(test_cats_dir)))
print('liczba obrazów testowych psów:', len(os.listdir(test_dogs_dir)))

# %% Budowa sieci neuronowej
# 3
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Input((150,150,3)))
# Zmiana głębokości map cech XX w kolejnych warstwach
# model.add(layers.Conv2D(XX, (3, 3), activation='relu'))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))

# kolejne warstwy
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# %% Kompilacja modelu poprzez binarną entropię krzyżową dla f. straty
# 4
from keras import optimizers
model.compile(loss='binary_crossentropy',
    optimizer='adam', #optimizers.RMSprop(learning_rate=1e-4),
    metrics=['acc'])

# %% Wstepna obrobka danych
# 5
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Przeskalowuje wszystkie obrazy o współczynnik 1/255.
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
  # Katalog docelowy.
    train_dir,
  # Zmienia rozdzielczość wszystkich obrazów na 150x150.
    target_size=(150, 150),
    batch_size=20,
  # Używamy funkcji binary_crossentropy w charakterze funkcji 
  #straty, a więc potrzebujemy binarnych etykiet.
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# %% Wartosci jednego z generatorów
# 6
for data_batch, labels_batch in train_generator:
    print('kształt danych wsadowych:', data_batch.shape)
    print('kształt etykiet danych wsadowych:', labels_batch.shape)
    break

# %% 7. Trenowanie 
# przekazujemy obiekty generatorów trenowania i walidacji oraz określamy liczbe epok

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator)
# %% zapis modelu do pliku
model.save(base_dir + 'cats_and_dogs_small_1.keras')

# %% 8 Wykresy 
# straty i dokładności pracy modelu podczas przetwarzania danych treningowych i walidacyjnych

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'r', label='Dokladnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Strata trenowania')
plt.plot(epochs, val_loss, 'r', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.legend()
plt.show()


# %% 9 Augmentacja danych
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# %% 10. kilka zmodyfikowanych obrazków
# Operacja importowania modułu zawierającego narzędzia przetwarzajace obrazy.
from tensorflow.keras.preprocessing import image
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# Wybieramy obraz do zmodyfikowania.
img_path = fnames[3]

# Wczytujemy obraz i zmieniamy jego rozdzielczość.
img = image.load_img(img_path, target_size=(150, 150))

# Zamieniamy obraz w tablicę Numpy o kształcie (150, 150, 3).
x = image.img_to_array(img)

# Zmieniamy kształt na (1, 150, 150, 3).
x = x.reshape((1,) + x.shape)

# Polecenie .flow() generuje wsady obrazów zmodyfikowanych w sposób losowy. 
# Pętla jest wykonywana w nieskończoność, a więc należy ją w pewnym momencie przerwać!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

# %% 11. Sieć neuronowa dla wersji z augmentacją

model = models.Sequential()
model.add(layers.Input((150,150,3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', #optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])

# %% 12. trenowanie sieci przy użyciu augmentacji danych i odrzucania

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Nie powinno się modyfikować danych walidacyjnych!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    # Katalog docelowy.
    train_dir,
    # Zmienia rozdzielczość wszystkich obrazów na 150x150.
    target_size=(150, 150),
    batch_size=128, # 32, 64, 256
    # Korzystamy z funkcji straty binarnej entropii krzyżowej, a więc potrzebujemy etykiet w formie binarnej.
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=128, # 32, 64, 256
    class_mode='binary')

history = model.fit(
    train_generator,
    epochs=50, # dla szybszych komputerów ustaw 100
    validation_data=validation_generator)

model.save(base_dir + 'cats_and_dogs_small_2.h5')


# %% 13.  wykresy parametrów modelu

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'r', label='Dokladnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Strata trenowania')
plt.plot(epochs, val_loss, 'r', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.legend()
plt.show()


