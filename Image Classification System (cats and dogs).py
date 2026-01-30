import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image


def load_and_preprocess_data(cats_dir,dogs_dir,img_size=128,test_split=0.2):
    print("Loading and preprocessing images...")

    images=[]
    labels=[]

    for file in os.listdir(cats_dir)[:1000]:  
        try:
            img_path=os.path.join(cats_dir, file)
            img=Image.open(img_path).convert('RGB')
            img=img.resize((img_size, img_size))
            img_array=np.array(img) / 255.0
            images.append(img_array)
            labels.append(0)  # 0 for cats
        except:
            continue
#
    for file in os.listdir(dogs_dir)[:1000]:  
        try:
            img_path=os.path.join(dogs_dir, file)
            img=Image.open(img_path).convert('RGB')
            img=img.resize((img_size, img_size))
            img_array=np.array(img) / 255.0
            images.append(img_array)
            labels.append(1)  # 1 for dogs
        except:
            continue

    images=np.array(images)
    labels=np.array(labels)

    indices=np.arange(len(images))
    np.random.shuffle(indices)
    images=images[indices]
    labels=labels[indices]

    # Split into train/test
    split_idx=int(len(images)*(1 - test_split))
    X_train, X_test=images[:split_idx], images[split_idx:]
    y_train, y_test=labels[:split_idx], labels[split_idx:]

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train,X_test,y_train,y_test


def build_cnn_model(input_shape=(128, 128, 3)):
    model=keras.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),

        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),

        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(1, activation='sigmoid', name='output')
    ])

    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    print("\nCompiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return history, model


def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")

    y_pred_prob=model.predict(X_test)
    y_pred=(y_pred_prob>0.5).astype(int)

    test_loss, test_acc=model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test Loss: {test_loss:.4f}")

    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cat', 'Dog'],
                yticklabels=['Cat', 'Dog'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

    return test_acc

def plot_training_history(history):
    fig, axes=plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def predict_single_image(model,image_path,img_size=128):
    try:
        img=Image.open(image_path).convert('RGB')
        img=img.resize((img_size, img_size))
        img_array=np.array(img) / 255.0
        img_array=np.expand_dims(img_array, axis=0)
        prediction=model.predict(img_array)[0][0]

        if prediction > 0.5:
            result="Dog"
            confidence=prediction
        else:
            result="Cat"
            confidence=1 - prediction

        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Prediction: {result} ({confidence:.2%} confidence)')
        plt.show()

        print(f"\nPrediction: {result}")
        print(f"Confidence: {confidence:.2%}")

        return result, confidence

    except Exception as e:
        print(f"Error: {e}")
        return None, None


def main():
    base_path="/content/drive/MyDrive/Project 3 Image Classification System/data/PetImages"
    cats_dir=os.path.join(base_path,"cats")
    dogs_dir=os.path.join(base_path,"dogs")

    if not os.path.exists(cats_dir):
        print(f"Error: Cats directory not found at {cats_dir}")
        return
    if not os.path.exists(dogs_dir):
        print(f"Error: Dogs directory not found at {dogs_dir}")
        return

    X_train, X_test, y_train, y_test = load_and_preprocess_data(cats_dir, dogs_dir)

    print("\nBuilding CNN model...")
    model=build_cnn_model()
    model.summary()

    history, trained_model=train_model(model, X_train, y_train, X_test, y_test, epochs=10)

    plot_training_history(history)
    evaluate_model(trained_model, X_test, y_test)

    print("\n" + "=" * 60)
    print("CLI PREDICTION DEMONSTRATION")
    print("=" * 60)

    sample_cat=os.path.join(cats_dir, os.listdir(cats_dir)[0])
    sample_dog=os.path.join(dogs_dir, os.listdir(dogs_dir)[0])

    print("\n1. Testing with a cat image:")
    predict_single_image(trained_model, sample_cat)

    print("\n2. Testing with a dog image:")
    predict_single_image(trained_model, sample_dog)

    model.save('/content/drive/MyDrive/cats_dogs_classifier.h5')
    print("\nModel will saved as 'cats_dogs_classifier.h5'")

#CLI 
cli_script='''
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py my_cat.jpg")
        sys.exit(1)

    image_path=sys.argv[1]

    # Loading model
    model=tf.keras.models.load_model('cats_dogs_classifier.h5')

    # Preprocess image
    img=Image.open(image_path).convert('RGB')
    img=img.resize((128, 128))
    img_array=np.array(img) / 255.0
    img_array=np.expand_dims(img_array, axis=0)

    prediction=model.predict(img_array)[0][0]

    if prediction>0.5:
        result="Dog"
        confidence=prediction
    else:
        result="Cat"
        confidence=1 - prediction

    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")

    return result

if __name__=="__main__":
    main()
'''

if __name__=="__main__":
    main()