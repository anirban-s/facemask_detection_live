{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Sequential models without an `input_shape` passed to the first layer cannot reload their optimizer state. As a result, your model isstarting with a freshly initialized optimizer.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:Sequential models without an `input_shape` passed to the first layer cannot reload their optimizer state. As a result, your model isstarting with a freshly initialized optimizer.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.07121758 0.92878246]]\n",
      "[[0.7020887 0.2979113]]\n",
      "[[0.2711236 0.7288764]]\n",
      "[[0.18163107 0.8183689 ]]\n",
      "[[0.6783364  0.32166365]]\n",
      "[[0.67956877 0.32043117]]\n",
      "[[0.06430177 0.9356983 ]]\n",
      "[[0.6570733  0.34292668]]\n",
      "[[0.01514394 0.98485607]]\n",
      "[[0.05174229 0.9482577 ]]\n",
      "[[0.4475952 0.5524049]]\n",
      "[[0.09039 0.90961]]\n",
      "[[0.24113125 0.75886875]]\n",
      "[[0.57367706 0.42632294]]\n",
      "[[0.41193774 0.5880622 ]]\n",
      "[[0.4118525  0.58814746]]\n",
      "[[0.4041504  0.59584963]]\n",
      "[[0.02484207 0.9751579 ]]\n",
      "[[0.032996 0.967004]]\n",
      "[[0.1157834 0.8842166]]\n",
      "[[0.15008254 0.8499175 ]]\n",
      "[[0.04638981 0.95361024]]\n",
      "[[0.07409164 0.9259084 ]]\n",
      "[[0.01874233 0.9812577 ]]\n",
      "[[0.11170114 0.88829887]]\n",
      "[[0.27046603 0.7295339 ]]\n",
      "[[0.02349319 0.9765069 ]]\n",
      "[[0.23203914 0.7679608 ]]\n",
      "[[0.38048416 0.6195159 ]]\n",
      "[[0.1527955 0.8472045]]\n",
      "[[0.02891565 0.9710843 ]]\n",
      "[[0.45923358 0.54076636]]\n",
      "[[0.23253414 0.7674659 ]]\n",
      "[[0.6253068  0.37469327]]\n",
      "[[0.75627285 0.24372716]]\n",
      "[[0.74428207 0.2557179 ]]\n",
      "[[0.9393244  0.06067566]]\n",
      "[[0.06677733 0.93322265]]\n",
      "[[0.31217223 0.68782777]]\n",
      "[[0.04602319 0.9539768 ]]\n",
      "[[0.07599518 0.9240048 ]]\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "from tensorflow.keras.models import load_model\n",
    "import tensorflow_hub as hub\n",
    "model = load_model('model/-mobilenetv2-Adam.h5', custom_objects={\"KerasLayer\": hub.KerasLayer})\n",
    "\n",
    "labels_dict = {0: 'withmask', 1: 'Withoutmask'}\n",
    "\n",
    "color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}\n",
    "\n",
    "size = 4\n",
    "webcam = cv2.VideoCapture(0)\n",
    "\n",
    "# load the xml file\n",
    "classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')\n",
    "\n",
    "while True:\n",
    "    (rval, im) = webcam.read()\n",
    "    im = cv2.flip(im, 1, 1)  # Flip to act as a mirror\n",
    "\n",
    "    # Resize the image to speed up detection\n",
    "    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))\n",
    "\n",
    "    # detect MultiScale / faces\n",
    "    faces = classifier.detectMultiScale(mini)\n",
    "\n",
    "    # Draw rectangles around each face\n",
    "    for f in faces:\n",
    "        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup\n",
    "        # Save just the rectangle faces in SubRecFaces\n",
    "        face_img = im[y:y+h, x:x+w]\n",
    "        resized = cv2.resize(face_img, (224, 224))\n",
    "        normalized = resized/255.0\n",
    "        reshaped = np.reshape(normalized, (1, 224, 224, 3))\n",
    "        reshaped = np.vstack([reshaped])\n",
    "        result = model.predict(reshaped)\n",
    "        print(result)\n",
    "\n",
    "        label = np.argmax(result, axis=1)[0]\n",
    "\n",
    "        cv2.rectangle(im, (x, y), (x+w, y+h), color_dict[label], 2)\n",
    "        cv2.rectangle(im, (x, y-40), (x+w, y), color_dict[label], -1)\n",
    "        cv2.putText(im, labels_dict[label], (x, y-10),\n",
    "                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)\n",
    "\n",
    "    # Show the image\n",
    "    cv2.imshow('Mask detection', im)\n",
    "    key = cv2.waitKey(10)\n",
    "    # if q key is press then break out of the loop\n",
    "    if key == ord('q'):\n",
    "        break\n",
    "# Stop video\n",
    "webcam.release()\n",
    "\n",
    "# Close all started windows\n",
    "cv2.destroyAllWindows()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
