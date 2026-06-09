# Final Model

After implementing and evaluating our model baseline the next step would be to implement a deeper CNN to improve image classification. Here we will incorporate multiple convolutional layers, pooling, more advances data augmentation parameters, callbakcs and visual aids like confusion matrix.
We want to increase accuray and reduce oevrfitting and overall achive a better performance in our model. 

## Data Augmentation
````
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.1),
    layers.RandomFlip("vertical")
])
````
To improve generalization and reduce overfitting we applied more hiperparameters during training like:
- Horizontal Flio
- Vertical Flip
- Random Rotation
- Random Zoom
- Random Translation
- Random Contrast

## Defining the model
````
def get_model_CNN_V2(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,

        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        
        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),

        layers.Dense(num_classes, activation="softmax")
    ])

    return model
model_cnn_v2 = get_model_CNN_V2(input_shape=(64, 64, 3), num_classes=num_classes)
model_cnn_v2.summary()
````
This "final" model was created as an improvment of the previous CNN architecture, with this one we increased the models ability tp exytact visual features from our dataset.
`data_augmentation` is defined before entering the convolutional layer because the images are randomly transformed through data augmentation (helps reduce overfitting).
<br>First convolutional layer `layers.Conv2D(32, 3, activation="relu", padding="same"),
layers.MaxPooling2D()`applies 32 filters of size 3 by 3 so it helps it learn visual patterns and it continues from 32, then 54, then 128 and finally 256 filters and with this one it learns even more complex visual representations.
<br>`layers.Flatten()` after the convolutional layers the features are STILL stored as multidimensional features, the flatten layer turns these features into a one-dimensional vector that can be processed by the connected layers.
<br>`layers.Dense(128, activation="relu")` with this layer we combine all of the previous features and learns from the connections between them before it makes the final prediction.
<br>`layers.Dense(num_classes, activation="softmax")` Here we have 1 neuron for each class, and since we have 18 classes the layer produces 18 outputs, with "softmax" makes the outputs into probabilities, the character with the highest probability will be the final prediction.
<br> 

## Compile the Model
[Compile the Model](#then-we-compile-the-model)


## Checkpoint Callback
After we defined the model we implemented a `callback` to automatically save the best version of the model during training.

```
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_best = ModelCheckpoint(
    filepath="best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=0
)
```
Basically here we save the best models accuracy during training and store it `filepath="best_model.keras"`, with this we can later try the model and make predictions with the best part of the model and when a better model is found during training it´s automatically updated. 
<br>We opted to `monitor="val_accuracy"` because it measures the model´s performance on the data that was not used for learning during that specific epoch.


## Training the Model
````
def train_model(model, train_img, train_labels):
    history = model.fit(
        train_img,
        train_labels,
        validation_split=0.2,
        epochs=25,
        batch_size=64,
        callbacks=[checkpoint_best]
    )
    return history

history = train_model(model_cnn_v2, train_img, train_labels)
````
This training function works like previous versions, the main thing to look at here is the `callbacks=[checkpoint_best]`, during the training this function monitored "val_accuracy" during every epoch and automatically saved the best performing model.


## Evaluating Test set

````
#Evaluamos el modelo con el set de prueba
test_loss, test_accuracy = model_cnn_v2.evaluate(test_img, test_labels)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
````
74/74 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - accuracy: 0.6188 - loss: 1.3175
Test loss: 1.3174875974655151
Test accuracy: 0.6188245415687561

<br> Here we test the model with totally new images, images that were not seen during the training phase to see how it will react, and looking at the result it had a "good" result.


## Evaluation Metrics
```
y_pred_probs = model_cnn_v2.predict(test_img)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = test_labels
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

accuracy = accuracy_score(test_labels, y_pred_classes)
precision = precision_score(test_labels, y_pred_classes, average="weighted")
recall = recall_score(test_labels, y_pred_classes, average="weighted")
f1 = f1_score(test_labels, y_pred_classes, average="weighted")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
```

Although accuracy is one of the most common metric, it does not always provide a complete picture of how the model is performing. To obtain a more valid and detailed evaluation we added more metrics:
- Accuraacy
- Precision
- Recall
- F1-Score

The model first generates a probability distribution for every image. the network provides 18 outputs data corresponds to the 18 classes in the dataset and the class that has the highest probability is chosen as the final prediction. 
<br>
```
Ace     → 0.92
Luffy   → 0.03
Zoro    → 0.02
```
The models prediction would be "Ace" because it has the highest probability.

- Accuracy tells us the percentage of the correctly classified images.
- Precision measures how many of the model´s positive predictions are actualy correct.
- Recall calculates how many of the actual class instances were correct.
- F1-Score is the combination of Precision and Recall into on powerful metric, it´s more useful when we have a multiclass classification.
<br>

We selected these metrics because we can relate them with the paper we chose, the same evaluation strategy was implemented to compare and ensure every model was understood.


## Mostramos las métricas de nuestro modelo
``
import pandas as pd

#Creamos un DataFrame para poder visualizar mejor las métricas de nuestro modelo
results_cnn_v2 = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        accuracy * 100,
        precision * 100,
        recall * 100,
        f1 * 100
    ]
})

results_cnn_v2["Value"] = results_cnn_v2["Value"].round(2)

results_cnn_v2
``



The results show that the improved CNN model achieved a strong improvment compared to our baseline models, precisión is just a little bit higher thatn accuracy so when de model predicts a charater some of those predictions are reliable, recall says that the model was also able to detect a good part of the real examples for each class. Then we have the F1-Score, which confirms that the model has a good balance between Precision and Recall.
<br>
It shows a hughe improvement compared to our first baseline models. 












## Training Curves
````
frame = pd.DataFrame(history.history)

train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(12,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(train_acc, label="Training", linewidth=2)
plt.plot(val_acc, label="Validation", linewidth=2)
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(train_loss, label="Training", linewidth=2)
plt.plot(val_loss, label="Validation", linewidth=2)
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()
````

To make sure we visualize and understand the behaviour of the model during the training process we made these graphics. 
- Accuracy vs Epochs
<br>Shows us how the models accuracy evolved during training, the training accuracy represents the performance on the images used for learning. The objective is for both curves to increase and stay close to each other, when the gap is big it usualy means there is overfitting involved.

- Loss vs Epochs
<br>Here it shows the prediction error during training, a good training process usualy shows a decrease in training loss, decrease of validation loss if validation loss starts to increase during training while training loss decreases it means overfitting may be ocurring.


## Confusion Matrix

````
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(test_labels, y_pred_classes)
fig , ax = plt.subplots(figsize=(8,8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, colorbar=True)
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.show()
````
With this Confusion Matrix we can see that the main diagonal representes the correctly classified images, if it has a strong diagonal it tells us that the model learned to distinguish between the different One Piece characters and cells outside the diagonal represente classification errors.
<br>This is important because while having accuracy it´s only a single performance value and with the confusion matrix it provides us a detailed view of model behaviour and allows us to answer several important questions like:

- Which classes are classified correctly most often?
- Which classes cause confusion?
- Are there specific characters that are harder to recognize?
<br> This is what makes it very valuable. 


## Saving the Model
````
model_cnn_v2.save("onepiece_model.keras")
print("Modelo guardado correctamente!")


"""
loaded_model = tf.keras.models.load_model(
    "onepiece_model.keras"
)
"""
````

Here is where we save the model, we save it because training a CNN can take time depending on the dataset, with this we save time and only has to be performed once.


## Image Classification Query System

````
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar el modelo guardado
model_query = tf.keras.models.load_model("best_model.keras")

# Función para preparar una imagen nueva
def cargar_imagen_query(image_path, img_size=(64, 64)):
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# Función para predecir una imagen
def predecir_imagen(image_path):
    img, img_array = cargar_imagen_query(image_path)

    predictions = model_query.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    predicted_class = class_names[predicted_index]

    print("Predicción:", predicted_class)
    print("Confianza:", confidence)

    plt.imshow(img)
    plt.title(f"Predicción: {predicted_class} | Confianza: {confidence:.2f}")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence

image_path = "/Users/arturosr/Desktop/OnePiece/OnePieceImage/Imagenes/OnePieceImagnesPrueba/05.png"

predecir_imagen(image_path)
````

After everything that was done the only thing left to do is to try the model, we load images outside the dataset and see how it performs. `model_query = tf.keras.models.load_model(
    "best_model.keras"
)` we load the best model that we saved so we dont have to retrain the whole CNN everytime we need to classify an image. 
<br>We have to process the images because images that are not from the dataset may have different dimensions, thats why we use: ` def cargar_imagen_query(image_path, img_size=(64, 64)):`, we have to prepare the images so the CNN can process them correctly.
<br>Then we load the new image, process them and generate a prediction with our CNN, it then displays the image with the prediction of the class and the "trust" it has on that prediction.






# References
- Alem, A., & Kumar, S. (2022). Deep Learning Models Performance Evaluations for Remote Sensed Image Classification. IEEE Access, 10, 111784–111793. https://doi.org/10.1109/ACCESS.2022.3215264
- Simonyan, K., & Zisserman, A. (2015).
Very Deep Convolutional Networks for Large-Scale Image Recognition.
International Conference on Learning Representations (ICLR).
- Mikołajczyk, A., & Grochowski, M. (2018).
Data augmentation for improving deep learning in image classification problem.
2018 International Interdisciplinary PhD Workshop (IIPhDW).
IEEE.

