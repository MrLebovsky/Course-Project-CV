from tkinter import *
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import cv2

Classifier = cv2.CascadeClassifier('FinnalClassifier/cascade.xml')

def select_classifier():
	
	global Classifier
	
	path = tkFileDialog.askopenfilename()
	if len(path) > 0:
		Classifier = cv2.CascadeClassifier(path)
		messagebox.showinfo('', 'Конфигурация классификатора успешно загружена!')
		
def select_image():
    # grab a reference to the image panels
    global panelA, panelB, image, path

	
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #  represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        image = Image.fromarray(image)

        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)

        # if the panels are None, initialize them
        if panelA is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)

        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelA.image = image


def detect_obj():

	global panelB, panelA
	
	paramA = txt.get()
	paramB = txt2.get()
	
	img = cv2.imread(path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	snowmans = Classifier.detectMultiScale(gray, float(paramA), int(paramB))
	
	for (x,y,w,h) in snowmans:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]

	if len(snowmans) > 0:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		img = ImageTk.PhotoImage(img)
		
		if panelB is None:
			panelB = Label(image=img)
			panelB.image = img
			panelB.pack(side="right", padx=10, pady=10)
		else:
			panelB.configure(image=img)
			panelB.image = img		
	else:
		messagebox.showinfo('', 'Заданные объекты не распознаны!')
	
# initialize the window toolkit along with the two image panels
window  = Tk()
panelA = None
panelB = None
image = None
path = ""
f_top = Frame(window)
f_bot = Frame(window)

window.title("8PI-81 Course Project. Detect an etalon object")
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI

lbl = Label(f_top, text="scaleFactor: ")
lbl.pack(side=LEFT)

txt = Entry(f_top,width=10)
txt.pack(side=LEFT)
txt.insert(0, "1.3")

lbl2 = Label(f_top, text="minNeighbors: ")
lbl2.pack(side=LEFT)

txt2 = Entry(f_top,width=10)
txt2.pack(side=LEFT)
txt2.insert(0, "3")

btn = Button(f_bot, text="Detect an objects!", command=detect_obj)
btn.pack(side=RIGHT, fill="both", expand="yes", padx="10", pady="5")

btn = Button(f_bot, text="Open image", command=select_image)
btn.pack(side=RIGHT, fill="both", expand="yes", padx="10", pady="5")

btn = Button(f_bot, text="Load classifier", command=select_classifier)
btn.pack(side=RIGHT, fill="both", expand="yes", padx="10", pady="5")

f_bot.pack()
f_top.pack()

# kick off the GUI
window.mainloop()