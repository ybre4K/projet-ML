# ============================================================
# PROJET 8 : DÉTECTION DE MARQUE DE VOITURE
# Script avec affichage interactif des graphiques.
# ============================================================

import tensorflow as tf
import os

# On importe Matplotlib de manière standard pour l'affichage interactif
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam

print(f"Version de TensorFlow : {tf.__version__}")

# ============================================================
# 1. CHEMINS ET CONFIGURATION
# ============================================================
chemin_train = 'dataset/Images/Train'
chemin_test = 'dataset/Images/Test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# ============================================================
# 2. PRÉPARATION DES DONNÉES
# ============================================================
print("\n--- Étape 1 : Préparation des données ---")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    chemin_train,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    chemin_test,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

nombre_classes = len(train_generator.class_indices)
print(f"\n{nombre_classes} classes détectées : {train_generator.class_indices}")

# ============================================================
# 3. CONSTRUCTION DU MODÈLE
# ============================================================
print("\n--- Étape 2 : Construction du modèle ResNet50 ---")
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
predictions = Dense(nombre_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ============================================================
# 4. ENTRAÎNEMENT DU MODÈLE
# ============================================================
print(f"\n--- Étape 3 : Début de l'entraînement ({EPOCHS} époques) ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    verbose=1
)
print("\n✅ Entraînement terminé.")

# ============================================================
# 5. SAUVEGARDE DU MODÈLE (ACTION CRITIQUE)
# ============================================================
print("\n--- Étape 4 : Sauvegarde du modèle ---")
# J'ai renommé en model_resnet50.h5 pour être cohérent avec le projet
model_path = 'model_resnet50.keras' 
model.save(model_path)
print(f"✅ Modèle sauvegardé dans '{model_path}'")


# ============================================================
# 6. ÉVALUATION (AFFICHAGE INTERACTIF DES GRAPHIQUES)
# ============================================================
print("\n--- Étape 5 : Affichage des graphiques d'évaluation ---")
plt.figure(figsize=(14, 6))

# Graphique de l'Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy (entraînement)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation)')
plt.title('Évolution de l\'Accuracy')
plt.xlabel('Époques')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Graphique de la Perte (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss (entraînement)')
plt.plot(history.history['val_loss'], label='Loss (validation)')
plt.title('Évolution de la Perte (Loss)')
plt.xlabel('Époques')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()

# AFFICHE LA FENÊTRE AU LIEU DE SAUVEGARDER
print("\nUne fenêtre avec les graphiques va s'afficher.")
print("Veuillez fermer cette fenêtre pour que le script se termine complètement.")
plt.show()

print("\n🎉🎉🎉 Script terminé après la fermeture de la fenêtre graphique. 🎉🎉🎉")