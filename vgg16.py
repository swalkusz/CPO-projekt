# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 00:40:47 2024

@author: szymo
"""

# %% 1. Import bibliotek

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import os
import matplotlib.pyplot as plt
import numpy as np

# %% 2.  Utworzenie instancji modelu VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

# %% 3. 

conv_base.summary()

# %% 4. dodanie modelu conv_base do modelu sekwencyjnego
# 4
# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
model1 = conv_base.output
model1 = Flatten(name="flatten")(model1)
model1 = Dense(256, activation='relu')(model1)
output_layer = Dense(1, activation='sigmoid')(model1)
model = Model(inputs=conv_base.input, outputs=output_layer)
model.summary()


# %% 5. Zamrażanie warstwy
# W pakiecie Keras sieć zamraża się przypisując wartość False atrybutowi trainable
print('Liczba tensorów wag poddawanych trenowaniu przed zamrożeniem bazy:', len(model.trainable_weights))
conv_base.trainable = False
print('Liczba tensorów wag poddawanych trenowaniu po zamrożeniu bazy:', len(model.trainable_weights))

# %%
# 10
BATCH_SIZE = 20
base_dir = 'C:/Users/szymo/Downloads/projekt/projekt/Nowy folder/cats_and_dogs_small'

# Katalogi podzbiorów (zbioru treningowego, walidacyjnego i testowego).
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    # Katalog docelowy.
    train_dir,
    # Rozdzielczość wszystkich obrazów zostanie zmieniona na 150x150.
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    # Korzystamy z funkcji straty w postaci binarnej entropii krzyżowej, a więc potrzebujemy etykiet w formie binarnej.
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=2e-5),
              metrics=['acc'])

history = model.fit(
    train_generator,
    # steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=40, # dodelowo 30 - 50
    validation_data=validation_generator,
    # validation_steps=validation_generator.samples // BATCH_SIZE
    )

# %% 11. Zapis modelu do pliku i wykresy

model.save('cats_and_dogs_small_3.keras')
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

# %% 12 Dostrajanie - podgląd architektury przed dostrajaniem
conv_base.summary()

# %% 13. Odmrażanie górnyh warstw
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


# %% 14. Dostrajanie sieci
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=2e-5),
              metrics=['acc'])

history = model.fit(
    train_generator,
    # steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=40, # dodelowo 30 - 50
    validation_data=validation_generator,
    # validation_steps=validation_generator.samples // BATCH_SIZE
    )

# %% 15. Zapis modelu i wykresy
model.save('cats_and_dogs_small_4.keras')
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

# %% 16. Sprawdzenie działania modelu na danych testowych
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=BATCH_SIZE,
    class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('dokładnosc podczas testowania:', test_acc)

# %% Załadowanie modelu z pliku
# o ile istnieje potrzeba
from tensorflow.keras.models import load_model
model = load_model('cats_and_dogs_small_4.keras')
model.summary()