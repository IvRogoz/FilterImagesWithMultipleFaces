import os
import cv2
import sys
from os import listdir
from os.path import isfile, join
from tkinter.filedialog import askdirectory
import tkinter as tk
import shutil

from facenet_pytorch import MTCNN

# Create face detector
# margin around the detected faces: 20px
# keep_all: return all detected faces
# post_process: get out images that look more normal to the human eye
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cuda:0')

root = tk.Tk()
root.overrideredirect(1)
root.withdraw()

def update_progress(progress, total, found):
	filled_length = int(round(100 * progress / float(total)))
	sys.stdout.write('\r [\033[1;34mPROGRESS\033[0;0m] [\033[0;32m{0}\033[0;0m]:{1}% : "Found:"{2}'.format('#' * int(filled_length/5), filled_length, found))
	if progress == total:sys.stdout.write('\n')
	sys.stdout.flush()

def test_dir(dir):
		if not os.path.exists(dir):
			os.makedirs(dir)
			print("created folder : ", dir)
		else:
			print(dir, "folder already exists.")

directory = askdirectory()
faces_dir = join(directory, './faces/')
nop_dir = join(directory, './nop/')

test_dir(faces_dir)
test_dir(nop_dir)

onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f)) & f.endswith(".jpg")]
i = 0
nop = 0
for n in range(0, len(onlyfiles)):
	current_file = join(directory, onlyfiles[n])
	try:
		image = cv2.imread(current_file)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# Detect faces in the image
		boxes, probs, _ = mtcnn.detect(gray, landmarks=True)
		count = 0
		for index, person in enumerate(boxes):
			person = person.astype(int)
			x,y,x1,y1 = person
			if probs[index] > 0.98 and (x1-x)>50: 
				count +=1
				cv2.rectangle(image, (x, y), (x1, y1), (0,155,255),2)
		if (count > 1):
			# uncomment to show found faces
			# cv2.imshow('frame',image)
			# cv2.waitKey(100)
			shutil.move(current_file, join(faces_dir, onlyfiles[n]))
			i += 1
		update_progress(n, len(onlyfiles), len(boxes))
	except Exception as e:
		#print()
		#print(e,"<>",current_file ) 
		shutil.move(current_file, join(nop_dir, onlyfiles[n]))
		nop += 1

print()
print("Moved:",i)
print("Errored:",nop)
