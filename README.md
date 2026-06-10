# Instituto Tecnológico y de Estudios Superiores de Monterrey Campus Querétaro
## Arturo Sánchez Rodríguez | A01275427

# OnePiece character image classification 

## Introduction
In this project we have an objective to prepare a dataset full of images of OnePiece characters for a future model image classification. The means to this project is to transform the original images into numerical data that can be used by a Machine Learning model or Deep Learning (both are good with dealing with images).
In this stage of the project we are working primarly with the selection of the dataset, image loading, train and test splitting, and the necessary preprocessing steps needed before building a classification model.

## Dataset
The dataset that we used in this project contains images of One Piece characters that are organized into folders and each folder represents a different class, which means that its a different character.
### The general structure of the dataset is as follows:
- OnePiece/Data/Data
-   Ace/
-   Akainu/
-   Brook/
-   Chopper/
-   Crocodile/
-   Franky/
-   Jinbei/
-   Kurohige/
-   Law/
-   Luffy/
-   Mihawk/
-   Nami/
-   Rayleigh/
-   Robin/
-   Sanji/
-   Shanks/
-   Usopp/
-   Zoro/

<br>The dataset contains a total of 11,737 images distributed across 18 classes (characters).
This dataset was selected because it contains a large number of labeled images that belong to different classes, and the objective of this project is to classify images of different characters organized into separate folders makes it possible to train a supervised learning model.
<br>You can download the Dataset here: [One Piece Image Classifier](https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier).


## Models

1. [Model 1 - MLP Baseline](Modelo_1.md)

2. [Model 2 - CNN](Modelo_2.md)

3. [Model 3 - Transfer Learning](Modelo_3_TL.md)

4. [Model 4 - Transfer Learning + Fine-Tuning](Modelo_4_TF_FT.md)


## Final Models comparison

The results show a consistent improvement on every stage of the project, the best overall score was with Transfer Learning and Fine Tuning reaching `71.64%` accuracy, f1-Score of `71.58%`. Fine-Tuning allowed us to adapt and use to our advantage the pre-trained features of the specific dataset (OnePiece), resulting in a better generalization of OnePiece images and its classification performance. 


The paper we selected reports a deeper CNN architecture and TL approach allowing and having a better image classification than shallow CNN models. This project follows the same path, the baseline CNN model had a low performance, Transfer Learning and Fine-Tuning kept on improving its accuracy, precision, recall and f1-score.
<br>
Even though there are very different datasets the findings and results that were obtained support the conclusions presented in the paper, deep architecture and TL techniques represents better feature extraction of image patterns and it improves overall classification performance.

![Model Comparison](Imagenes/ComparacionFinal.png)



# Selected Paper
The selected paper evaluates the performance of different deep learning architectures for image classification jut like our case og image classification with 18 classes.
This paper was selected because it follows a very strategic methodology and it is very simmilar to the one used in this project:
- Image classification
- Convolutional Neural Networks
- Accuracy 
- Recall
- F1-score
- Confusión matrix
Even though the datasets are different, this paper gives us a very useful reference of comparison and evaluating and improvising CNN-based image classification models.

<br>
Alem, A., & Kumar, S. (2022).
Deep Learning Models Performance Evaluations for Remote Sensed Image Classification.
IEEE Access, 10, 111784–111793.


# Image Classification Query System

````
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class_names = np.load("class_names.npy", allow_pickle=True) #Cargamos los nombres de las clases para la predicción

# Cargar el modelo guardado
model_query = tf.keras.models.load_model("best_model_tl_ft.keras")#Cargamos el modelo que guardamos con el mejor rendimiento para hacer las predicciones

# Función para preparar una imagen nueva
def cargar_imagen_query(image_path, img_size=(64, 64)): 
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array
#Cargamos la imagen que queremos que el modelo haga predicción y la preparamos para que pueda procesarla)

# Función para predecir una imagen 
def predecir_imagen(image_path):
    img, img_array = cargar_imagen_query(image_path)

    predictions = model_query.predict(img_array)#Hacemos la predicción con el modelo cargado utilizando la imagen que preparamos

    predicted_index = np.argmax(predictions[0])#Obtenemos el índice de la clase con la mayor probabilidad
    confidence = np.max(predictions[0])#Obtenemos la confianza de la predicción 

    predicted_class = class_names[predicted_index]#Sacamos el nombre de la clase de la predicción

    print("Predicción:", predicted_class)
    print("Confianza:", confidence)

    plt.imshow(img)
    plt.title(f"Predicción: {predicted_class} | Confianza: {confidence:.2f}")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence

#La ruta de la imagen que queremos predecir
image_path = "/Users/arturosr/Desktop/OnePiece/OnePieceImage/Imagenes/OnePieceImagnesPrueba/02.png"

predecir_imagen(image_path)
````

After everything that was done the only thing left to do is to try the model, we load images outside the dataset and see how it performs. `model_query = tf.keras.models.load_model(
    "best_model.keras"
)` we load the best model that we saved so we dont have to retrain the whole CNN everytime we need to classify an image. 
<br>We have to process the images because images that are not from the dataset may have different dimensions, thats why we use: ` def cargar_imagen_query(image_path, img_size=(64, 64)):`, we have to prepare the images so the CNN can process them correctly.
<br>Then we load the new image, process them and generate a prediction with our CNN, it then displays the image with the prediction of the class and the "trust" it has on that prediction.

![Queries Example](Imagenes/QueriesEj.png)





## Conclusion
Throughout this project, we explored differetn approaches for the main problem of image classification using a dataset full of One Piece characters, containing 18 different classes that represent 18 different characters. The workflow of starting with a simple model like the MLP and CNN helped us understand the dificulty of image classification and show how simple and non reliable a basic architecture has, then the next step would be to apply more parameters we can alter and manage to increase accuracy.
<br> 
Each stage of this project keeps on improving the performance of each model, the best results we had was with Transfer Learning + Fine-Tuning, with results like aproximately 71.64% accuracy and 71.58% F1-Score, which is a huge improvement. We can say and compare to our paper that deeper architectures and pretrained models can learn visual patterns better than simple CNN models, we share a conclusion with our paper where Transfer Learning and deeper networks will achieve better image classification results.
<br>
Overall the project followed a workflow which looks for improvment in each and every stage of the machine learning, including dataset preparation, preprocessing, model implementation, evaluation and refinement to keep searching for the best model possible.


# References
- One Piece Image Classifier Dataset.
Kaggle.
https://www.kaggle.com/datasets
- TensorFlow Developers.
TensorFlow Documentation.
https://www.tensorflow.org/
- Keras Developers.
Keras Documentation.
https://keras.io/
- Alem, A., & Kumar, S. (2022).
Deep Learning Models Performance Evaluations for Remote Sensed Image Classification.
IEEE Access, 10, 111784–111793.
https://doi.org/10.1109/ACCESS.2022.3215264
- Simonyan, K., & Zisserman, A. (2015).
Very Deep Convolutional Networks for Large-Scale Image Recognition.
International Conference on Learning Representations (ICLR).
https://arxiv.org/abs/1409.1556
- Mikołajczyk, A., & Grochowski, M. (2018).
Data Augmentation for Improving Deep Learning in Image Classification Problem.
2018 International Interdisciplinary PhD Workshop (IIPhDW).
IEEE.
https://doi.org/10.1109/IIPHDW.2018.8388338
- Tan, M., & Le, Q. V. (2019).
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
Proceedings of the 36th International Conference on Machine Learning (ICML).
https://arxiv.org/abs/1905.11946
- Artificial Intelligence Course Notes and Examples.
  TensorFlow MNIST Example.
  Data Augmentation Example.
  Transfer Learning Example.
  Evaluation and Refinement Example.
  Tecnológico de Monterrey, Campus Querétaro.
  2026.




