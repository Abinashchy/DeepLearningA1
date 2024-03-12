import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# Load the Fashion-MNIST dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()

# Define class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Plot one sample image for each class
plt.figure(figsize=(15, 15))

for i in range(len(class_names)):
    # Find the index of the first occurrence of the class in the labels
    index = np.where(y_train == i)[0][0]
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[index], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')

plt.show()