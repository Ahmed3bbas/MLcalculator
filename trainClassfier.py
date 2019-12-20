from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from collections import Counter


class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, Saved_Weights_Path=None):
        # Initialize the Model
        model = Sequential()

        # First CONV => RELU => POOL Layer
        model.add(Conv2D(20, 5, 5, border_mode="same", input_shape=(depth, height, width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))

        # Second CONV => RELU => POOL Layer
        model.add(Conv2D(50, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))

        # Third CONV => RELU => POOL Layer
        # Convolution -> ReLU Activation Function -> Pooling Layer
        model.add(Conv2D(100, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))

        # FC => RELU layers
        #  Fully Connected Layer -> ReLU Activation Function
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Using Softmax Classifier for Linear Classification
        model.add(Dense(total_classes))
        model.add(Activation("softmax"))

        # If the saved_weights file is already present i.e model is pre-trained, load that weights
        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)
        return model

CNN = CNN()

# --------------------------------- EOC ------------------------------------
import numpy as np
import cv2
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from data.data import load_data

# Parse the Arguments
save_weights = "pretrained/cnn_weights.h5"
model_name = 'pretrained/model.h5'
Saved_Weights_Path = None

if not os.path.exists('pretrained'):
    os.makedirs('pretrained')


# Read/Download MNIST Dataset
print('Loading Dataset...')

X_train, Y_train, X_test, Y_test = load_data()
# Divide data into testing and training sets.
train_img, train_labels, test_img, test_labels = X_train, Y_train, X_test, Y_test

# Now each image rows and columns are of 28x28 matrix type.
img_rows, img_columns = 28, 28

# Transform training and testing data to 10 classes in range [0,classes] ; num. of classes = 0 to 9 = 10 classes

total_classes = 13		# 0 to 9 labels + - *

#each code of matrix have a value like of class like [0,1,.....,+,*,...]

encoder = LabelEncoder()
tra = encoder.fit_transform(Y_train)
train_labels = to_categorical(tra)
tes = encoder.fit_transform(Y_test)
test_labels = to_categorical(tes)
target_labels = [i for i in encoder.classes_]

(valX,test_img, valY, test_labels) = train_test_split( test_img , test_labels , test_size= 0.5)


# Defing and compile the SGD optimizer and CNN model
print('\n Compiling model...')
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
clf = CNN.build(width=28, height=28, depth=1, total_classes=total_classes, Saved_Weights_Path= Saved_Weights_Path)
clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
print("Count of digits in dataset", Counter(Y_train))

# Initially train and test the model; If weight saved already, load the weights using arguments.
b_size = 128		# Batch size
num_epoch = 10		# Number of epochs
verb = 1			# Verbose

# If weights saved and argument load_model; Load the pre-trained model.
if not os.path.exists(save_weights):
	print('\nTraining the Model...')
	H = clf.fit(train_img, train_labels,validation_data=(valX, valY ), batch_size=b_size, epochs=num_epoch,verbose=verb)
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, num_epoch), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, num_epoch), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, num_epoch), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, num_epoch), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
 	
	predictions = clf.predict(test_img, batch_size=b_size)
	print(classification_report(test_labels.argmax(axis=1),predictions.argmax(axis=1),target_names=target_labels))
	# Evaluate accuracy and loss function of test data
	print('Evaluating Accuracy and Loss Function...')
	loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
	print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

	
# Save the pre-trained model.
if not  os.path.exists(save_weights):
	print('Saving weights to file...')
	clf.save_weights(save_weights, overwrite=True)
	clf.save(model_name)
 

# Show the images using OpenCV and making random selections.
for num in np.random.choice(np.arange(0, len(test_labels)), size=(5,)):
	# Predict the label of digit using CNN.
	probs = clf.predict(test_img[np.newaxis, num])
	prediction =[i for i in mapped if mapped[i][np.argmax(probs)] == 1]#probs.argmax(axis=1)
	#p = prediction
	#prediction = list(mapped.keys()).index(p)
	# Resize the Image to 100x100 from 28x28 for better view.
	image = (test_img[num][0] * 255).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(prediction[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
  
	# Show and print the Actual Image and Predicted Label Value
	print('Predicted Label: {}, Actual Value: {}'.format(prediction,Y_test[num]))
	cv2.imshow(image)
	cv2.waitKey(0)

#---------------------- EOC ---------------------
