# face-recognition

## Team members:
Kariuki

## Problem definition:
Recognition and identification of persons in the picture/video.

## Problem motivation:
Today, more and more modern security systems use some form of identification to control access to company employees. The solution to the problem can be implemented in bank security systems as one sees employee controls.
And the detection of people in the picture has become a massive problem for most social networks.
Based on the given examples of the application of detection and identification of persons, it can be seen that the system is currently being used for various problems.

## Data set:
I will manually generate the PID data set based on the example.

## Methodology:
The Python programming language and its OpenCV library will be used for the realization of the project. Using the CascadeClassifier and a set of points used for face detection link, the person's face will be detected.

Based on the data set that was previously generated, a neural network will be trained using the cv2.face.traing mechanism in order to obtain a prediction of persons in the image/video.

## Evaluation model:
For the evaluation model of recognition and identification of persons, he would use: precision, recall and f1-score.
