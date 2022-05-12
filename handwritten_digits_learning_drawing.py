import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

#load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#show example dataset image
plt.imshow(x_train[0])

#normalize data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 10)

#get validation loss
val_loss, val_acc = model.evaluate(x_test, y_test) 


drawing = False

def draw(event, x, y, flags, param):
    global drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img1, (x, y), 15, (255, 0, 0), -1)

#create blank canvas
img1 = np.zeros((512, 512, 1,), np.uint8)
cv2.namedWindow("image")
cv2.setMouseCallback("image", draw)
#drawing
while(1):
    cv2.imshow("image", img1)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('p'):
        img2 = img1 / 255.0
        img2 = cv2.resize(img2, (28, 28), cv2.INTER_AREA)
        img2 = img2.reshape(1, 28, 28)        
        pr = model.predict_classes(img2)
        print(pr)
    elif k == ord('c'):
        img1 = np.zeros((512, 512, 1), np.uint8)
    elif k == ord('s'):
        img2 = cv2.resize(img1, (28, 28), cv2.INTER_AREA)
        cv2.imwrite('my_fig.png', img2)
cv2.destroyAllWindows() 
