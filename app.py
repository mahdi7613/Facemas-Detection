from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
import numpy as np
import cv2
from deeplearning import face_mask_prediction


class VideoCapture(qtc.QThread):
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.run_flag = True
        
    def run(self):
        cap = cv2.VideoCapture(0)
        
        while self.run_flag:
            ret , frame = cap.read()
            prediction_img = face_mask_prediction(frame)
            
            if ret == True:
                self.change_pixmap_signal.emit(prediction_img)
                
        cap.release()
        
    def stop(self):
        self.run_flag = False
        self.wait()

class mainWindows(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(qtg.QIcon('./images/icon.png'))
        self.setWindowTitle('Face Mask Recogonition Software')
        self.setFixedSize(600,600)
        
        # Adding Widgets
        label = qtw.QLabel('<h2>Face Mask Recogonition Application</h2>')
        self.cameraButton = qtw.QPushButton('Open Camera',clicked=self.cameraButtonClick,checkable = True)
        
        # Screen
        self.screen = qtw.QLabel()
        self.img = qtg.QPixmap(600,480)
        self.img.fill(qtg.QColor('darkGrey'))
        self.screen.setPixmap(self.img)
        
        
        # Layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.cameraButton)
        layout.addWidget(self.screen)
        
        self.setLayout(layout)
        
        self.show()
    
    
    def cameraButtonClick(self):
        print('clicked')
        status = self.cameraButton.isChecked()
        if status == True:
            self.cameraButton.setText('Close Camera')
            
            # Open Camera
            self.capture = VideoCapture()
            self.capture.change_pixmap_signal.connect(self.updateImage)
            self.capture.start()
            
        elif status == False:
            self.cameraButton.setText('Open Camera')
            self.capture.stop()
            
    @qtc.pyqtSlot(np.ndarray)  
    def updateImage(self,image_array):
        rgb_image = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
        h,w, ch = rgb_image.shape
        bytes_per_line = ch*w
        # conver into 
        convertedImage = qtg.QImage(rgb_image.data,w,h,bytes_per_line,qtg.QImage.Format_RGB888)
        scaledImage = convertedImage.scaled(600,480,qtc.Qt.KeepAspectRatio)
        qt_image = qtg.QPixmap.fromImage(scaledImage)
        
        # update to screen
        self.screen.setPixmap(qt_image)
        
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mw = mainWindows()
    sys.exit(app.exec())