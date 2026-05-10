import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ============================================================
# 1. CONFIGURATION (ADAPTÉE À VOS FICHIERS)
# ============================================================
# --- CORRECTION ICI ---
# Chemin vers le modèle sauvegardé (avec la bonne extension .keras)
MODEL_PATH = 'model_resnet50.keras'

# Taille des images (doit être la même que pour l'entraînement)
IMG_SIZE = (224, 224)

# Dictionnaire des classes (dans l'ordre alphabétique)
CLASSES = {
    0: 'audi',
    1: 'lamborghini',
    2: 'mercedes'
}

# ============================================================
# 2. FONCTION DE PRÉDICTION
# ============================================================
def predict_car_brand(model, image_path):
    if not os.path.exists(image_path):
        print(f"❌ ERREUR : Le fichier image '{image_path}' n'a pas été trouvé.")
        return

    print(f"\nChargement et préparation de l'image : {image_path}")
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array_batch = np.expand_dims(img_array, axis=0)

    print("Prédiction en cours...")
    predictions = model.predict(img_array_batch, verbose=0)
    
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    class_name = CLASSES[predicted_class_index]
    
    print("\n" + "="*50)
    print("          RÉSULTAT DE LA PRÉDICTION")
    print("="*50)
    print(f"➡️  Marque prédite : {class_name.upper()}")
    print(f"🎯 Confiance      : {confidence:.2%}")
    print("-" * 50)
    print("Probabilités pour chaque classe :")
    for i, prob in enumerate(predictions[0]):
        print(f"  - {CLASSES[i]:12s} : {prob:.4f}")
    print("="*50 + "\n")

    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.title(f"Prédiction : {class_name.upper()}\nConfiance : {confidence:.2%}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

# ============================================================
# 3. SCRIPT PRINCIPAL
# ============================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERREUR FATALE : Le fichier modèle '{MODEL_PATH}' est introuvable.")
    else:
        try:
            print(f"Chargement du modèle '{MODEL_PATH}'...")
            model = load_model(MODEL_PATH)
            print("✅ Modèle chargé avec succès !\n")

            while True:
                image_path = input("Entrez le chemin de l'image (ex: teste_voiture.jpg) ou 'exit' : ").strip().strip('"')

                if image_path.lower() == 'exit':
                    break
                
                predict_car_brand(model, image_path)

            print("\nProgramme de test terminé.")

        except Exception as e:
            print(f"❌ Une erreur est survenue lors du chargement du modèle : {e}")
