import sys
from PyQt6 import QtCore,QtGui
import cv2
import numpy
import os, time
import os.path
from glob import glob
import re

fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'database'
fn_train = 'trainer.yml'
(im_width, im_height) = (300, 300)
faceCascade = cv2.CascadeClassifier(fn_haar)
recognizer = cv2.createLBPHFaceRecognizer()
TRAINING = 'training'
TESTING = 'testing'


def proba():
    recognizer = cv2.createLBPHFaceRecognizer()
    # Part 1: Create fisherRecognizer
    print('Training...')
    # Create a list of images and a list of corresponding names
    os.remove(fn_train)
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(fn_dir):
         for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(fn_dir, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
                id += 1

    (images, labels) = [numpy.array(lis) for lis in [images, labels]]

    print(str(numpy.array(images)))
    recognizer.train(images, numpy.array(labels))
    recognizer.save(fn_train)

def Name(nameUser):
    print("Name Verification: "+str(nameUser))
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(fn_dir):
        for subdir in dirs:
            print("The name is: "+str(subdir))
            if(subdir == nameUser):
                return False
            else:
                return True

def addUser(nameUser):
    count = 0
    size = 5
    print("Popped in here to take a headshot:" + str(nameUser))
    path = os.path.join(fn_dir, str(nameUser))
    if not os.path.isdir(path):
        os.mkdir(path)
        fn_haar=cv2.CascadeClassifier(fn_haar)
        webcam=cv2.VideoCapture(0)


    print ("--Taking pictures--","--Give some expressions--")


    while count < 100:
        (rval, im) =webcam.read()
        im = cv2.flip(im, 1, 0)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gray, (gray.shape[1] / 5, gray.shape[0] / 5))
        faces = fn_haar.detectMultiScale(mini)
        faces = sorted(faces, key=lambda x: x[2])
        if faces:
            face_i = faces[0]
            (x, y, w, h) = [v * 5 for v in face_i]
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                   if n[0]!='.' ]+[0])[-1] + 1
            cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(im, str(nameUser), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            time.sleep(0.38)        
            count += 1	    	
            cv2.imshow('OpenCV', im)
            key = cv2.waitKey(10)
        if key == 27:
            break
    print(str(count) + " images have been taken and saved to " + str(nameUser) +" folder in database ")
    cv2.destroyWindow("OpenCV")

def predictionDetectFace(images, labels, image):
    recognizer.train(images, numpy.array(labels))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.25, 6,flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    while True:
        for (x, y, w, h) in faces:
            cv2.rectangle(images,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (im_width, im_height))
            prediction = recognizer.predict(face_resize)
            if prediction[1] < 150:
                print(str(prediction[0]) + " : " + str(prediction[1]))
                return 1, 0, 0, True
            if prediction[1] < 250:
                #print("<200: " + str(prediction[1]))
                return 0, 0, 1, False
            else:
                return 0, 1, 0, False

def predictionImages(path, images):
    (images, labels, names, id) = ([], [], {}, 0)
    count = 0
    tp = 0
    fp = 0
    tn = 0
    user = ""
    for filename in os.listdir(path):
        label = 1
        images.append(cv2.imread(path+'/'+filename, 0))
        labels.append(int(label))
        tpp, fpp, tnn, us = predictionDetectFace(images, labels, images)
        tp += tpp
        tn += tnn
        fp += fpp
        if us:
            userName = path.split('/')
            user = userName[1]
            print(user)
        count += 1
        if 9 < count:
            return tp, tn, fp, user

def searchName():
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(fn_dir):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(fn_dir, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label)) 
                id += 1
                return images, labels, names, id


class ControlWindow(QtGui.QWidget):
    def __init__(self,*args):
        super(QtGui.QWidget, self).__init__()

        self.nameUser = ""
        self.images = ""
        self.save = False


        self.start_button = QtGui.QPushButton('Turn on the camera')
        self.start_button.clicked.connect(self.startCapture)
        
        self.start_img = QtGui.QPushButton('Upload image')
        self.start_img.clicked.connect(self.startImage)

        self.update_button = QtGui.QPushButton('Start/Stop update')
        self.update_button.clicked.connect(self.updateUser)

        # ------ Modification ------ #
        self.capture_button = QtGui.QPushButton('Add new user')
        self.capture_button.clicked.connect(self.addNewUser)

        self.training_button = QtGui.QPushButton('Train')
        self.training_button.clicked.connect(self.training)

        self.testtt = QtGui.QPushButton('Test')
        self.testtt.clicked.connect(self.test)
        # ------ Modification ------ #

        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.start_img)
        vbox.addWidget(self.update_button)

        # ------ Modification ------ #
        vbox.addWidget(self.capture_button)
        vbox.addWidget(self.training_button)
        vbox.addWidget(self.testtt)
        # ------ Modification ------ #

        self.setLayout(vbox)
        self.setWindowTitle('Control Panel')
        self.setGeometry(100,100,200,200)
        self.show()

        self.recognizerMetod()
        print("recognizer: " + str(recognizer))

    def test(self):
        print("Validity Testing")
        fileName = QtGui.QFileDialog.getOpenFileName(self, 'OpenFile')
        imagePath = str(fileName)
        print("Please wait a moment...")
        font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
        self.images = cv2.imread(imagePath)
        images, labels, names, id = searchName()
        count = 0;
        ime = ""
        (full_res, true_positive, false_positive, true_negative) = (0.0, 0.0, 0.0, 0.0)
        while count < len(names):
            path = fn_dir + "/" + names[count]
            print("How many copies do you want:" + str(path))
            tp, tn, fp, user = predictionImages(path, self.images)
            full_res += (tp + fp + tn)
            true_positive += tp
            false_positive += fp
            true_negative += tn
            if user != "":
                print("When he's here")
                ime = user
            print(ime)
            count += 1

        precision = ((true_positive)/(true_positive + false_positive))
        recall = ((true_positive)/(true_negative + true_positive))
        f1 = 2*((precision * recall) / (precision + recall))
        print("true: positive " + str(true_positive) + "| true negative: " + str(true_negative) + "| false positive: " + str(false_positive) + "| full_res: " + str(full_res))
        
        print("The person in the picture is " + str(ime))
        print('%s - %.2f' % ("Precison", precision * 100.0) + "%")
        print('%s - %.2f' % ("Recall", recall * 100) + "%")
        print('%s - %.2f' % ("F1-score", f1 * 100) + "%")

def updateUser(self):
    print("Update the knowledge base")
    if self.save:
        self.save = False
    else:
        self.save = True


def recognizerMetod(self):
        recognizer.load('trainer.yml')
        print(str("Memorize"))

def startCapture(self):
        print("Start the recognition")
        viewCam(self.recognizer)
        labels, names, id = searchName() 
        fn_haar = cv2.CascadeClassifier(fn_haar)
        webcam=cv2.VideoCapture(0)        
while True:
            (_, img)=webcam.read()
            img = cv2.flip(img, 1, 0)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces =fn_haar.detectMultiScale(gray, 1.25, 6,flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

            #Iterating through rectangles of detected faces
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x, y),(x+w,y+h),(0,255,0),thickness=2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))
                # Try to recognize the face
                prediction = recognizer.predict(face_resize)
                print(str(prediction[0]) + " - " + str(prediction[1]))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if prediction[1] < 60:
                    if prediction[1] > 70:
                        print("Was the person questioned?")
                    else:
                        cv2.putText(img,'%s - %.0f' % (Name[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                        if self.save:
                            cv2.imshow('Face', face_resize)
                            nameUser = Name[prediction[0]]
                            path = os.path.join(fn_dir, str(nameUser))
                            if not os.path.isdir(path):
                                os.mkdir(path)
                            pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                                    if n[0]!='.' ]+[0])[-1] + 1
                            cv2.imwrite('%s/%s.png' % (path, pin), face_resize)


                else:
                            cv2.putText('Am not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                            cv2.imshow('OpenCV', img)
                            if cv2.waitKey(27) & 0xFF == ord('q'):
                                    break
                            cv2.destroyWindow("Video")

                def startImage(self):
                    print("Start recognizing a person from an image")
                    fileName = QtGui.QFileDialog.getOpenFileName(self, 'OpenFile')
                    print("fileName: " + str(fileName))
                    imagePath = str(fileName)

                    font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
                    self.images = cv2.imread(imagePath)
                    self.viewImage(self.images)

                def addNewUser(self):
                            print("Adding a new user to database...")
                            odg = True

                            self.nameUser, result = QtGui.QInputDialog.getText(self, "Creating a base name!","Enter the name of the person you want to add to the system")
                            if result:
                                correctly = correctlyName(self.nameUser)
                                if correctly:
                                    print("The person you want to add %s!" % self.nameUser)
                                    addUser(self.nameUser)
                                    print("You have added a new user.");
                                    proba()
                                    print("New user has been added.")
                                    self.recognizerMetod()
                                else:
                                    print("This name is taken please try with another name")
                                    while odg:
                                        self.nameUser, result = QtGui.QInputDialog.getText(self, "Creating a base name the name is taken please try another one")
                                        if result:
                                            correctly = correctlyName(self.nameUser)
                                            if correctly:
                                                print("The person you want to add %s!" % self.nameUser)
                                                odg = False;
                                            else:
                                                print("This name is taken please try with another name")
                                                odg = True;
                                            addUser(self.nameUser)
                                            proba()
                                            self.recognizerMetod()
                def training(self):
                        print("Neural network training...")
                        proba()
                        print("Training finished.")
                        self.recognizerMetod()

                def viewImage(self, images):
                    images, labels, names, id = searchName()
                    font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,3,1,0,4)
                while True:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = faceCascade.detectMultiScale(gray, 1.25, 6,flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
                    od = True
                    while True:
                        for(x, y, w, h) in faces:
                            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),thickness=2)
                            face = gray[y:y + h, x:x + w]
                            face_resize = cv2.resize(face, (im_width, im_height))
                            prediction = recognizer.predict(face_resize)
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            print(str(prediction))
                            if prediction[1] < 70:
                                cv2.cv.PutText(cv2.cv.fromarray(image), '%s - %.0f' % (str(names[prediction[0]]), prediction[1]), (x,y+h),font, 255)
                            else:
                                cv2.putText(image,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                                    
                            if od:
                                cv2.imshow("add", face_resize)
                                nameUser, result = QtGui.QInputDialog.getText(self, "Add a new user","Enter the name of the new user")
                                od = result
                            if result:
                                path = os.path.join(fn_dir, str(nameUser))
                            if not os.path.isdir(path):
                                os.mkdir(path)
                            count = 0
                        while count < 10:
                            pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                            if n[0]!='.' ]+[0])[-1] + 1
                            cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
                            count +=1
                            od = False
                            self.training()
                            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                            cv2.imshow('Video', image)
                            if cv2.waitKey(27) & 0xFF == ord('q'):
                                break
                            cv2.destroyWindow("Video")
                            
                            if __name__ == '__main__':
                                        app = QtGui.QApplication(sys.argv)
                                        window = ControlWindow()
                                        sys.exit(app.exec_())