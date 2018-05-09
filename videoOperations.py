import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
from PIL import Image
import sys
import glob
from shutil import copyfile
import mmap
from PIL import ImageFont
from PIL import ImageDraw 

def video_to_image(video_folder_path):
    all_videos = os.listdir(video_folder_path)
    for video_name in all_videos:
        if video_name.endswith(".mp4"):
            if not os.path.exists(str(video_folder_path)+'' +str(str(video_name).split('.')[0])+'/'):
             os.makedirs(str(video_folder_path)+'' +str(str(video_name).split('.')[0])+'/')
            save_image_path=str(video_folder_path)+ ''+str(str(video_name).split('.')[0])+'/'
            convert_to_images(str(video_folder_path)+str(video_name),save_image_path,os.getcwd())
def save_to_video():
	output_path='/home/kafein/Projeler/Face Detection/tum_yuzler/result_images/'
	list_files = sorted(get_file_names(output_path), key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
	img0 = cv2.imread(os.path.join(output_path,'/home/beyhan/Face Detection/tum_yuzler/result_images/1517613902.jpg'))
	#cv2.imshow('asd',img0)
	height , width , layers =  img0.shape

	# fourcc = cv2.cv.CV_FOURCC(*'mp4v')
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	#fourcc = cv2.cv.CV_FOURCC(*'XVID')
	videowriter = cv2.VideoWriter(output_video_file,fourcc, frame_rate, (width,height))
	for f in list_files:
		print("saving..." + f)
		img = cv2.imread(os.path.join(output_path, f))
		videowriter.write(img)
	videowriter.release()
	cv2.destroyAllWindows()
def get_file_names(search_path):
	for (dirpath, _, filenames) in os.walk(search_path):
		for filename in filenames:
			yield filename#os.path.join(dirpath, filename)
def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
def convert_to_images(input_video_file,img_path,base_dir):
	cam = cv2.VideoCapture(input_video_file)
        print('Start save frames!')
	counter=0
	counter2=0
	sola,solu,saga,sagu=0,0,0,0
	while True:
		counter=counter+1
		flag, frame = cam.read()
		if flag:
                                if counter2%25==0:
                                        print(counter2)
					cv2.resize(frame, (96, 96), interpolation = cv2.INTER_AREA)
					new_image_name=os.path.join(img_path, str(counter2) + '.jpg')
                                        print(new_image_name)
                                        frame = cv2.resize(frame, (720, 720)) 
					cv2.imwrite(new_image_name,frame)
                                        image_part=os.path.join(str(img_path),'parts_of_image')
                                        if not os.path.exists(image_part):
                                         os.makedirs(image_part)
					os.chdir(base_dir)
                                        sola,solu,saga,sagu=frame_split(new_image_name,image_part,img_path,str(counter2) + '.jpg',sola,solu,saga,sagu,base_dir)
				counter2=counter2+1
		else:
				break
		if cv2.waitKey(1) == 27:
			break
			# press esc to quit
	cv2.destroyAllWindows()
def frame_split(src_img,dst_path,base_path,base_image_name,sola,solu,saga,sagu,home_dir):
	if not os.path.exists(dst_path) or not os.path.isfile(src_img):
	    print 'Not exists', src_img, img_path
	    sys.exit(1)

	im = Image.open(src_img)
	im_w, im_h = im.size
        w,h=int(im_w/2), int(im_h/2)
        print(w)
        print(h)
	print 'Image width:%d height:%d  will split into (%d %d) ' % (im_w, im_h, w, h)
	w_num, h_num = int(im_w/w), int(im_h/h)

	for wi in range(0, w_num):
	    for hi in range(0, h_num):
		box = (wi*w, hi*h, (wi+1)*w, (hi+1)*h)
		piece = im.crop(box)
		tmp_img = Image.new('L', (w, h), 255)
		tmp_img.paste(piece)
                if wi==0 and hi==0: 
		 print('SolUst')
                 is_there_any_person=save_part('SolUst',tmp_img,dst_path)
		 solu=int(solu)+int(is_there_any_person)
		 img = Image.open(os.path.join(dst_path,'SolUst.png'))
		 draw = ImageDraw.Draw(img)
		 font = ImageFont.truetype(os.path.join(home_dir,"myfont.ttf"), 32)
		 draw.text((0, 0),str(sola),(255),font=font)
		 img.save(os.path.join(dst_path,'SolUst.png'))
		 print('kayit etti mi ')
                if wi==0 and hi==1: 
		 print('SolAlt')
                 is_there_any_person=save_part('SolAlt',tmp_img,dst_path)
		 sola=int(sola)+int(is_there_any_person)
		 img = Image.open(os.path.join(dst_path,'SolAlt.png'))
		 draw = ImageDraw.Draw(img)
		 font = ImageFont.truetype(os.path.join(home_dir,"myfont.ttf"), 32)
		 draw.text((0, 0),str(sola),(255),font=font)
		 img.save(os.path.join(dst_path,'SolAlt.png'))
                if wi==1 and hi==0: 
		 print('SagUst')
                 is_there_any_person=save_part('SagUst',tmp_img,dst_path)
		 sagu=int(sagu)+int(is_there_any_person)
		 img = Image.open(os.path.join(dst_path,'SagUst.png'))
		 draw = ImageDraw.Draw(img)
		 font = ImageFont.truetype(os.path.join(home_dir,"myfont.ttf"), 32)
		 draw.text((0, 0),str(sagu),(255),font=font)
		 img.save(os.path.join(dst_path,'SagUst.png'))
                if wi==1 and hi==1: 
		 print('SagAlt')
                 is_there_any_person=save_part('SagAlt',tmp_img,dst_path)
		 saga=int(saga)+int(is_there_any_person)
		 img = Image.open(os.path.join(dst_path,'SagAlt.png'))
		 draw = ImageDraw.Draw(img)
		 font = ImageFont.truetype(os.path.join(home_dir,"myfont.ttf"), 32)
		 draw.text((0, 0),str(saga),(255),font=font)
		 img.save(os.path.join(dst_path,'SagAlt.png'))
        os.chdir(dst_path)
        SolUst = Image.open('SolUst.png')
	SolAlt = Image.open('SolAlt.png')
	SagAlt = Image.open('SagAlt.png')
	SagUst = Image.open('SagUst.png')

	img1 = append_images([SolUst, SagUst], direction='horizontal')
	img2 = append_images([SolAlt, SagAlt], direction='horizontal')
	final = append_images([img1, img2], direction='vertical')
	final.save(str(base_path)+str(base_image_name))
	return sola,solu,saga,sagu
	#merge_images(dst_path)
def save_part(name,tmp_img,dst_path):
	new_part_image_name=str(name)+'.png'
	img_path = os.path.join(dst_path, str(new_part_image_name))
	tmp_img.save(img_path)
	is_there_any_person=0
	is_there_any_person=detection(img_path,dst_path,new_part_image_name)
        return is_there_any_person
def detection(img_path,dst_path,new_part_image_name):
	os.chdir('/home/beyhan/OtomasyonProjeErsin/darknet')
	os.system('./darknet detect cfg/yolov3.cfg yolov3.weights '+str(img_path) +'> '+dst_path+'/cikti.txt')
	is_there_any_person=0
	f = open(dst_path+'/cikti.txt')
	s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
	if s.find('insan') != -1:
	    is_there_any_person=1
	    print('Insan bulundu!')
	else:
	    print('Insan bulunamadi!')
	copyfile('/home/beyhan/OtomasyonProjeErsin/darknet/predictions.png', os.path.join(dst_path,new_part_image_name))
	return is_there_any_person

def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im

