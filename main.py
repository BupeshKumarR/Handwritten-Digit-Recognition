'''

Handwritten Digit Recognizer using MNIST

A Beginner level OpenCV program using Python Deep Learning

This project can be scaled up to build an application that reads handwritten texts in different human fonts
and transforms them into digital information.

Steps:
1. Load Data from MNIST
2.Normalize

'''



import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

train_new_model = True

if train_new_model:
    mnist = tf.keras.datasets.mnist

#1.Load data from MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()


#2. Normalize
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
#3. Model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#4.Model Fit
    model.fit(X_train, y_train, epochs=5)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(loss)
    print(accuracy)

    model.save('handwritten_digits.model')
else:

    model = tf.keras.models.load_model('handwritten_digits.model')


def predict(model, img):
    imgs = np.array([img])
    res = model.predict(imgs)
    index = np.argmax(res)
    return str(index)


# left mouse click handler
startInference = False


def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference



# threshold slider handler
threshold = 100


def on_threshold(x):
    global threshold
    threshold = x

# the opencv display loop
def start_cv(model):
    global threshold
    cap = cv2.VideoCapture(0)
    frame = cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', 150, 255, on_threshold)
    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    while True:
        ret, frame = cap.read()

        if (startInference):

            # frame counter for showing text
            frameCount += 1

            # black outer frame
            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # apply threshold
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)

            # get central image
            resizedFrame = thr[240 - 75:240 + 75, 320 - 75:320 + 75]
            background[240 - 75:240 + 75, 320 - 75:320 + 75] = resizedFrame

            # resize for inference
            iconImg = cv2.resize(resizedFrame, (28, 28))

            # pass to model predictor
            res = predict(model, iconImg)

            # clear background
            if frameCount == 5:
                background[0:480, 0:80] = 0
                frameCount = 0

            # show text
            cv2.putText(background, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(background, (320 - 80, 240 - 80), (320 + 80, 240 + 80), (255, 255, 255), thickness=3)

            # display frame
            cv2.imshow('background', background)
        else:
            # display normal video
            cv2.imshow('background', frame)

        # cv2.imshow('resized', resizedFrame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()







# main function
def main():

    print("starting cv...")

    # show opencv window
    start_cv(model)


# call main
if __name__ == '__main__':
    main()
