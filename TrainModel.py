import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Check shapes of the loaded data
# x_train.shape, y_train.shape, x_test.shape, y_test.shape

# Preprocess the images

# Normalize pixel values to [0,1] range
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

# Expand image dimensions to (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Create the neural network model
model = Sequential()

# Add Convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))

# Flatten the output for Dense layers
model.add(Flatten())

# Add Dropout layer for regularization
model.add(Dropout(0.25))

# Add Dense layer for output
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

# Early Stopping to prevent overfitting
# Sử dụng model.compile() để biên dịch mô hình với optimizer 'adam', hàm mất mát là categorical cross-entropy (do đây là bài toán phân loại nhiều lớp), và sử dụng chỉ số độ chính xác ('accuracy') để đánh giá hiệu suất mô hình trong quá trình huấn luyện.
es = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1)

# Model Checkpoint to save the best model
# Thiết lập để dừng sớm quá trình huấn luyện nếu độ chính xác trên tập validation không cải thiện (delta=0.01) sau một số lượng epoch quy định (patience=4).
mc = ModelCheckpoint('./mnist.h5', monitor='val_acc', verbose=1, save_best_only=True)

# Use callbacks during training
# Lưu mô hình có độ chính xác tốt nhất trên tập validation vào một file './mnist.h5'.
callbacks = [es, mc]

# Train the model
with tf.device('/device:GPU:0'):  # Training with GPU
    history = model.fit(x_train, y_train, epochs=50, validation_split=0.3, callbacks=callbacks)

print("The model has been successfully trained")

# Save the trained model
model.save('mnist.h5')
