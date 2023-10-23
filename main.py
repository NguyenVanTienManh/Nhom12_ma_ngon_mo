import random
import cv2


# Function to Keep track of all the faces present
def draw_face_rectangles(image,face_coordinates):
  track = 0
  # Looping through all the faces present in the image or frame
  for (x,y,w,h) in face_coordinates:
    color=(random.randint(0,255),random.randint(0,255),random.randint(0,255))
    cv2.rectangle(image,(x,y),(x+w,y+h),color,3)


trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Assigining the trained face data to a variable

# Importing the image fam.jpg from the directory and assigning it to the varible img
img = cv2.imread('anh.jpg')

grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # COnerts the coloured image to a gray scale to be easily decipherable by the program

face_coordinates = trained_face_data.detectMultiScale(
  grayscale)  # Asssigning the nested list of cordinates of the faces detected to a variable face_coordinates

draw_face_rectangles(img,face_coordinates)  # Calling the tracker function on face coordinates

width, height = 800, 600  # Kích thước mà bạn muốn
img = cv2.resize(img, (width, height))

cv2.imshow('Face Detector', img) # Hiển thị ảnh với kích thước đã điều chỉnh
cv2.waitKey()
cv2.destroyAllWindows()


print("Code Completed!")