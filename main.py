from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist
from keras.preprocessing import image
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

item_train_data = []
item_test_data = []
(train_x_data, train_y_data), (test_x_data,
                               test_y_data) = fashion_mnist.load_data()
for item_train in train_x_data:
    item_train_data.append(item_train.flatten())
for item_test in test_x_data:
    item_test_data.append(item_test.flatten())
item_train_data = np.array(item_train_data)
item_test_data = np.array(item_test_data)
print(item_test)
item_mnist_classifier = OneVsRestClassifier(
    LogisticRegression(verbose=1, max_iter=10))
item_mnist_classifier.fit(item_train_data, train_y_data)
conf_matrix = confusion_matrix(
    test_y_data, item_mnist_classifier.predict(item_test_data))
print("Confusion_matrix:")
print(conf_matrix)
sns.heatmap(conf_matrix)
print('The output score is: %s' %
      item_mnist_classifier.score(item_test_data, test_y_data))
plt.show()

item_mnist_classifier = LogisticRegression(
    verbose=1, max_iter=6, multi_class="multinomial", solver="sag")
item_mnist_classifier.fit(item_train_data, train_y_data)
conf_matrix = confusion_matrix(
    test_y_data, item_mnist_classifier.predict(item_test_data))
sns.heatmap(conf_matrix)
item_mnist_classifier.score(item_test_data, test_y_data)
plt.show()
print('The output score is:  %s' %
      item_mnist_classifier.score(item_test_data, test_y_data))
pickle.dump(item_mnist_classifier, open(
    'item_mnist_classifier.model', 'wb'))
item_mnist_classifier__from_file = pickle.load(
    open('item_mnist_classifier.model', 'rb'))
conf_matrix = confusion_matrix(
    test_y_data, item_mnist_classifier__from_file.predict(item_test_data))

sns.heatmap(conf_matrix)
print('The output score is:  %s' %
      item_mnist_classifier__from_file.score(item_test_data, test_y_data))
plt.show()

img = image.load_img('file.png', target_size=(
    28, 28), grayscale=True, color_mode="grayscale")
x = image.img_to_array(img)
y = x.flatten().reshape(1, -1)
print(item_mnist_classifier__from_file.predict(y))
