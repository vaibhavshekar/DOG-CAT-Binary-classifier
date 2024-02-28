import cv2
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

# import cv2
# import imghdr
# data_dir = 'data'
# image_exts = ['jpeg','jpg', 'bmp', 'png']
# for image_class in os.listdir(data_dir):
#     for image in os.listdir(os.path.join(data_dir, image_class)):
#         image_path = os.path.join(data_dir, image_class, image)
#         try:
#             img = cv2.imread(image_path)
#             tip = imghdr.what(image_path)
#             if tip not in image_exts:
#                 print('Image not in ext list {}'.format(image_path))
#                 os.remove(image_path)
#         except Exception as e:
#             print('Issue with image {}'.format(image_path))

data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])


# 0 is cat, 1 is dog

# x_values = []
# count = 0
# for x, _ in data:
#     x_values.append(x)
#     count += 1
# x_values_tensor = tf.concat(x_values, axis=0)
#
# print(x_values_tensor)
#
# sum_of_pixel_values = tf.reduce_sum(x_values_tensor)
# sum_of_pixel_values = sum_of_pixel_values/count
#
# # Print the sum of pixel values
# print("Sum of all pixel values across all images:", sum_of_pixel_values.numpy())

data = data.map(lambda x,y: (x/255, y))

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape = (256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
# model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))) # doesnt help with validation_accuracy
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))) #tuning
# model.add(BatchNormalization()) #tuning - reduced precision and accuracy signifcantly
model.add(Dense(1,activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])

model.summary()


hist = model.fit(train,epochs = 20, validation_data = val)
# model.fit(train,epochs = 20, validation_data = val)

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)

    # Convert probabilities to binary predictions using threshold
    yhat_binary = (yhat > 0.5).astype(int)

    # Update state of metrics with ground truth labels and binary predictions
    pre.update_state(y, yhat_binary)
    re.update_state(y, yhat_binary)
    acc.update_state(y, yhat_binary)

# Get final results after updating state with all batches
precision_result = pre.result()
recall_result = re.result()
accuracy_result = acc.result()

print("Precision:", precision_result)
print("Recall:", recall_result)
print("Accuracy:", accuracy_result)

img_data = cv2.imread('cat_test.jpg')
resize = tf.image.resize(img_data,(256,256))
predct_val = model.predict(np.expand_dims(resize/255, 0))
print('prediction value: ', predct_val)
if predct_val < 0.5:
    print(f'its a cat')
else:
    print(f'its a dog')
