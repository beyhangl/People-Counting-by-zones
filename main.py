#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import sys
import glob
from PIL import Image
import videoOperations as videoOp

video_folder_path='/home/beyhan/OtomasyonProjeErsin/'

videoOp.video_to_image(video_folder_path)
#videoOp.frame_split('/home/beyhan/OtomasyonProjeErsin/test_video/0.jpg','/home/beyhan/OtomasyonProjeErsin/test_video/parts_of_image')
