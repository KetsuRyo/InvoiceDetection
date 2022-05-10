# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import json
import cv2
import time
import argparse
import fileinput
import sys
import os
import glob
import threading
import shutil
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import gc
from os import listdir



__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import re
# det
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# rec
import math
import traceback
import paddle
import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
import imutils
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from numba import cuda

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logger = get_logger()

app = Flask(__name__)

CORS(app)
txt = str()
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def detect_read_files():
    filenames = [x.split(".jpg")[0] for x in os.listdir('yolov5/Invoice_Normalization/output')]
    classes = pd.read_csv('yolov5/classes.txt', index_col=None, header=None)
   #建立資料夾 tagged_image/各個classes
    return filenames, classes

def detect_to_cv2_bbox(bbox, img_shape):
    bbox = [(bbox[1]-bbox[3]/2)*img_shape[0], (bbox[1]+bbox[3]/2)*img_shape[0], (bbox[0]-bbox[2]/2)*img_shape[1], (bbox[0]+bbox[2]/2)*img_shape[1]]
    bbox = [round(x) for x in bbox]
    return bbox

def detect_export_tagged_image():
    filenames, classes = detect_read_files()
    for filename in filenames:
        img = cv2.imdecode(np.fromfile(r'yolov5/Invoice_Normalization/output/%s.jpg' %filename, dtype=np.uint8), -1)                
        labels = pd.read_csv('runs/detect/exp/labels/' + filename + '.txt', index_col=0, header=None, sep=' ')
        img_shape = img.shape
        ids = np.zeros_like(classes)
        for row in labels.iterrows():
            ids[row[0]] += 1
            bbox = detect_to_cv2_bbox(row[1].to_list(), img_shape)
            img_class = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            export_filename = './outputImage/tagged_image/' + classes[0][row[0]] + '/' + f"{filename}_{ids[row[0]]}.jpg"
            cv2.imencode('.jpg', img_class)[1].tofile(export_filename)

def replacement(file, previousw, nextw):
    for line in fileinput.input(file, inplace=1):
        line = line.replace(previousw, nextw)
        sys.stdout.write(line)


def normalize():
    label_file_path = os.path.join( "runs", "detect", "exp")
    file_path = os.path.join( "yolov5", "InputImage" )

    image_save_path = os.path.join("yolov5", "Invoice_Normalization", "output")
    img_filenames = glob.glob(os.path.join(file_path, "*"))

    scale_coefficient = 1500
    angle_coefficient = 3.525
    trim_dx1 = scale_coefficient*0.01
    trim_dy1 = scale_coefficient*0.03
    trim_dx2 = int(scale_coefficient*0.7)
    trim_dy2 = int(scale_coefficient*1.2)

    for img_filename in img_filenames:
        # label and image save filename
        filename = os.path.splitext(os.path.basename(img_filename))[0]
        label_filename = os.path.join(label_file_path, "labels", filename) + ".txt"

        # read image and label info
        img = cv2.imdecode(np.fromfile(img_filename, dtype=np.uint8), -1)
        label = np.loadtxt(label_filename)

        print(label)


        if (len(label) == 2):
            # normalization
            img_shape = img.shape
            x1 = int(img_shape[1]*label[0][1])
            y1 = int(img_shape[0]*label[0][2])
            x2 = int(img_shape[1]*label[1][1])
            y2 = int(img_shape[0]*label[1][2])
            #cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3
                     
            dx = x1 - x2
            dy = y1 - y2

            angle = np.arctan(dy/dx)*180/np.pi - angle_coefficient
            length = (dx**2 + dy**2)**0.5
            scale_ratio = scale_coefficient / length
            size = (int(img.shape[1]*scale_ratio),
                    int(img.shape[0]*scale_ratio))

            M = cv2.getRotationMatrix2D((y1, x1), angle, 1)
            img = cv2.warpAffine(
                img, M, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

            trim_x1 = int(y1*scale_ratio - trim_dx1)
            trim_x2 = trim_x1 + trim_dx2
            trim_y1 = int(x1*scale_ratio - trim_dy1)
            trim_y2 = trim_y1 + trim_dy2
            print(trim_x1, trim_x2, trim_y1, trim_y2)
            print(img.shape)
            img = img[trim_x1:trim_x2, trim_y1:trim_y2]
      

        # export result
        img_save_filename = os.path.join(
            image_save_path, filename) + ".jpg"
        cv2.imencode(".jpg", img)[1].tofile(img_save_filename)
        print('COMPLETE :', filename)
        
        
        
def main(args,targetfile):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []
    filename_list = []
    labels = {}
    with open(targetfile, "a") as f:
        f.write("file""\t""prediction")
        f.write("\n")
    # warmup 2 times
    if args.warmup:
        img = np.random.uniform(0, 255, [32, 320, 3]).astype(np.uint8)
        for i in range(2):
            res = text_recognizer([img] * int(args.rec_batch_num))

    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
        filename_list.append(os.path.basename(image_file))
    try:
        rec_res, _ = text_recognizer(img_list)

    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    results = 0
    for ino in range(len(img_list)):
        predict = rec_res[ino][0]
        filename = filename_list[ino]
        logger.info("Img:{}, Pred:{}".format(filename, predict))
        with open(targetfile, "a") as f:
            filename = filename.replace('_[1]', '' )
            filename = filename.replace('_[2]', '' )
            filename = filename.replace('_[3]', '' )
            filename = filename.replace('_[4]', '' )
            filename = filename.replace('_[5]', '' )
            filename = filename.replace('_[6]', '' )
            filename = filename.replace('_[7]', '' )
            filename = filename.replace('_[8]', '' )
            filename = filename.replace('_[9]', '' )
            filename = filename.replace('_[10]', '' )
            if predict =='':
                f.write("{}\t{}".format(filename, 'none'))
                f.write("\n")
            else:
                predict= predict.replace('\t','')
                predict= predict.replace('　','')
                f.write("{}\t{}".format(filename, predict))
                f.write("\n")
    mypath = "./yolov5/InputImage"
    files = listdir(mypath)
    replacement(targetfile,' ','')
    replacement(targetfile,':','')
    replacement(targetfile,'：','')
    replacement(targetfile,'司','')
    replacement(targetfile,'元','')
    replacement(targetfile,'公','公司')
    replacement(targetfile,'.jpg','')
    for k in files:
        with open(targetfile) as f:
            d = str(k).replace('.png', '' )
            g = d.replace('.jpg','')
            g = g.replace('.JPG','')
            g = g.replace('.PNG','')
            g = g.replace('.bmp','')
            if g not in f.read():
                 with open(targetfile, "a") as ww:
                        ww.write("{}\t{}".format(g, 'cannotpredict'))
                        ww.write("\n")

    if args.benchmark:
        text_recognizer.autolog.report()
        gc.collect()




def get_TaxType():
    script = 'python3 ./taxtypeCNN/predict-multiple.py  -m ./taxtypeCNN/traffic_sign.model   -i ./outputImage/tagged_image/TaxType -s '
    result_success = subprocess.run(script, shell=True)
    return

class TextRecognizer(object):
    def __init__(self, args):
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char
        }
        if self.rec_algorithm == "SRN":
            postprocess_params = {
                'name': 'SRNLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "RARE":
            postprocess_params = {
                'name': 'AttnLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == 'NRTR':
            postprocess_params = {
                'name': 'NRTRLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        elif self.rec_algorithm == "SAR":
            postprocess_params = {
                'name': 'SARLabelDecode',
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char
            }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'rec', logger)
        self.benchmark = args.benchmark
        self.use_onnx = args.use_onnx
        if args.benchmark:
            import auto_log
            pid = os.getpid()
            gpu_id = utility.get_infer_gpuid()
            self.autolog = auto_log.AutoLogger(
                model_name="rec",
                model_precision=args.precision,
                batch_size=args.rec_batch_num,
                data_shape="dynamic",
                save_path=None,  #args.save_log_path,
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id if args.use_gpu else None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=0,
                logger=logger)

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        if self.rec_algorithm == 'NRTR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # return padding_im
            image_pil = Image.fromarray(np.uint8(img))
            img = image_pil.resize([100, 32], Image.ANTIALIAS)
            img = np.array(img)
            norm_img = np.expand_dims(img, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            return norm_img.astype(np.float32) / 128. - 1.

        assert imgC == img.shape[2]
        imgW = int((32 * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if w is not None and w > 0:
                imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0:img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length):

        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]

        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            self.srn_other_inputs(image_shape, num_heads, max_text_length)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)

    def resize_norm_img_sar(self, img, image_shape,
                            width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape
        h = img.shape[0]
        w = img.shape[1]
        valid_ratio = 1.0
        # make sure new_width is an integral multiple of width_divisor.
        width_divisor = int(1 / width_downsample_ratio)
        # resize
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        if imgW_min is not None:
            resize_w = max(imgW_min, resize_w)
        if imgW_max is not None:
            valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
            resize_w = min(imgW_max, resize_w)
        resized_image = cv2.resize(img, (resize_w, imgH))
        resized_image = resized_image.astype('float32')
        # norm 
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        resize_shape = resized_image.shape
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, 0:resize_w] = resized_image
        pad_shape = padding_im.shape

        return padding_im, resize_shape, pad_shape, valid_ratio

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()
        if self.benchmark:
            self.autolog.times.start()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm != "SRN" and self.rec_algorithm != "SAR":
                    norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                    max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "SAR":
                    norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                        img_list[indices[ino]], self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    valid_ratio = np.expand_dims(valid_ratio, axis=0)
                    valid_ratios = []
                    valid_ratios.append(valid_ratio)
                    norm_img_batch.append(norm_img)
                else:
                    norm_img = self.process_image_srn(
                        img_list[indices[ino]], self.rec_image_shape, 8, 25)
                    encoder_word_pos_list = []
                    gsrm_word_pos_list = []
                    gsrm_slf_attn_bias1_list = []
                    gsrm_slf_attn_bias2_list = []
                    encoder_word_pos_list.append(norm_img[1])
                    gsrm_word_pos_list.append(norm_img[2])
                    gsrm_slf_attn_bias1_list.append(norm_img[3])
                    gsrm_slf_attn_bias2_list.append(norm_img[4])
                    norm_img_batch.append(norm_img[0])
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            if self.benchmark:
                self.autolog.times.stamp()

            if self.rec_algorithm == "SRN":
                encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = np.concatenate(
                    gsrm_slf_attn_bias1_list)
                gsrm_slf_attn_bias2_list = np.concatenate(
                    gsrm_slf_attn_bias2_list)

                inputs = [
                    norm_img_batch,
                    encoder_word_pos_list,
                    gsrm_word_pos_list,
                    gsrm_slf_attn_bias1_list,
                    gsrm_slf_attn_bias2_list,
                ]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = {"predict": outputs[2]}
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = {"predict": outputs[2]}
            elif self.rec_algorithm == "SAR":
                valid_ratios = np.concatenate(valid_ratios)
                inputs = [
                    norm_img_batch,
                    valid_ratios,
                ]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs[0]
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(
                            input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs[0]
            else:
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors,
                                                 input_dict)
                    preds = outputs[0]
                else:
                    self.input_tensor.copy_from_cpu(norm_img_batch)
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    if len(outputs) != 1:
                        preds = outputs
                    else:
                        preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            if self.benchmark:
                self.autolog.times.end(stamp=True)
        return rec_res, time.time() - st

def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    elif engine and model.trt_fp16_input != half:
        LOGGER.info('model ' + (
            'requires' if model.trt_fp16_input else 'incompatible with') + ' --half. Adjusting automatically.')
        half = model.trt_fp16_input

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def main2(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_recognizer = TextRecognizer(args)
    valid_image_file_list = []
    img_list = []
    filename_list = []
    labels = {}
    with open("./Note2.txt", "a") as f:
        f.write("file""\t""prediction")
        f.write("\n")
    # warmup 2 times
    if args.warmup:
        img = np.random.uniform(0, 255, [32, 320, 3]).astype(np.uint8)
        for i in range(2):
            res = text_recognizer([img] * int(args.rec_batch_num))

    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
        filename_list.append(os.path.basename(image_file))
    try:
        rec_res, _ = text_recognizer(img_list)

    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()
    results = 0
    for ino in range(len(img_list)):
        predict = rec_res[ino][0]
        filename = filename_list[ino]
        logger.info("Img:{}, Pred:{}".format(filename, predict))
        with open("./Note2.txt", "a") as f:
            filename = filename.replace('_[1]', '' )
            filename = filename.replace('_[2]', '' )
            filename = filename.replace('_[3]', '' )
            filename = filename.replace('_[4]', '' )
            filename = filename.replace('_[5]', '' )
            filename = filename.replace('_[6]', '' )
            filename = filename.replace('_[7]', '' )
            filename = filename.replace('_[8]', '' )
            filename = filename.replace('_[9]', '' )
            filename = filename.replace('_[10]', '' )
            if predict =='':
                f.write("{}\t{}".format(filename, 'none'))
                f.write("\n")
            else:
                predict= predict.replace('\t','')
                predict= predict.replace('　','')
                f.write("{}\t{}".format(filename, predict))
                f.write("\n")
    mypath = "./yolov5/InputImage"
    files = listdir(mypath)
    replacement("./Note2.txt",' ','-')
    replacement("./Note2.txt",'.jpg','')
    for k in files:
        with open('./Note2.txt') as f:
            d = str(k).replace('.png', '' )
            g = d.replace('.jpg','')
            g = g.replace('.JPG','')
            g = g.replace('.PNG','')
            g = g.replace('.bmp','')
            if g not in f.read():
                 with open("./Note2.txt", "a") as ww:
                        ww.write("{}\t{}".format(g, 'cannotpredict'))
                        ww.write("\n")
        


        
    if args.benchmark:
        text_recognizer.autolog.report()
        gc.collect()
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def removefile():
    os.remove("InvoiceNumber.txt")
    os.remove("Buyer.txt")
    os.remove("BuyerTaxIDNumber.txt")
    os.remove("Note3.txt")
    os.remove("Note2.txt")
    os.remove("Tax.txt")
    os.remove("NetAmount.txt")
    os.remove("TotalAmount.txt")
    os.remove("ChineseAmount.txt")
    os.remove("TaxType.txt")
    os.remove("Date.txt")
    os.remove("SellerTaxIDNumber.txt")
    os.remove("Month.txt")
    os.remove("Year.txt")
    os.remove("InvoiceNumber2.txt")
    os.remove("Buyer2.txt")
    os.remove("BuyerTaxIDNumber2.txt")
    os.remove("Tax2.txt")
    os.remove("NetAmount2.txt")
    os.remove("TotalAmount2.txt")
    os.remove("ChineseAmount2.txt")
    os.remove("Date2.txt")
    os.remove("SellerTaxIDNumber2.txt")
    os.remove("Month2.txt")
    os.remove("Year2.txt")       
    os.remove("InvoiceNumber.json")
    os.remove("Buyer.json")
    os.remove("BuyerTaxIDNumber.json")
    os.remove("Note.json")
    os.remove("Tax.json")
    os.remove("NetAmount.json")
    os.remove("TotalAmount.json")
    os.remove("ChineseAmount.json")
    os.remove("TaxType.json")
    os.remove("Date.json")
    os.remove("SellerTaxIDNumber.json")
    os.remove("Month.json")
    os.remove("Year.json")   
    os.remove("yolo_identified_time.json")
    os.remove("OCR_time.json")   
    
    
def deleteAll():
    shutil.rmtree('./outputImage/tagged_image')
    shutil.rmtree('./yolov5/InputImage')
    shutil.rmtree('./runs')
    shutil.rmtree('./yolov5/Invoice_Normalization/output')
    os.mkdir('./yolov5/Invoice_Normalization/output')
    os.mkdir('./yolov5/InputImage')
    os.mkdir('./outputImage/tagged_image')
    os.mkdir('./outputImage/tagged_image/InvoiceNumber')
    os.mkdir('./outputImage/tagged_image/Buyer')
    os.mkdir('./outputImage/tagged_image/BuyerTaxIDNumber')
    os.mkdir('./outputImage/tagged_image/Date')
    os.mkdir('./outputImage/tagged_image/Note')
    os.mkdir('./outputImage/tagged_image/TaxType')
    os.mkdir('./outputImage/tagged_image/NetAmount')
    os.mkdir('./outputImage/tagged_image/SellerTaxIDNumber')
    os.mkdir('./outputImage/tagged_image/Tax')
    os.mkdir('./outputImage/tagged_image/TotalAmount')
    os.mkdir('./outputImage/tagged_image/ChineseAmount')
    os.mkdir('./outputImage/tagged_image/Year')
    os.mkdir('./outputImage/tagged_image/Month')

def process_text_to_json(targetfile,colomn,jsonfile):
    with open(targetfile) as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.split()
            dataJsoos ={
                "name": line[0],
                colomn: line[1]
            }
  
            with open(jsonfile, "a") as outfile:
                json.dump(dataJsoos, outfile)
                outfile.write("\n")

def process_text_to_json_ChineseAmount(targetfile,colomn,jsonfile):
    with open(targetfile) as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.split()
            dataJsoos ={
                "name": line[0],
                colomn: "%s元" %line[1]
            }
  
            with open(jsonfile, "a") as outfile:
                json.dump(dataJsoos, outfile)
                outfile.write("\n")                
                
                
            
def makeRecordFile():
    textOrigin = open("InvoiceNumber.txt","w+")
    textOrigin.close()
    textOrigin = open("Buyer.txt","w+")
    textOrigin.close()
    textOrigin = open("BuyerTaxIDNumber.txt","w+")
    textOrigin.close()
    textOrigin = open("Note2.txt","w+")
    textOrigin.close()
    textOrigin = open("Tax.txt","w+")
    textOrigin.close()
    textOrigin = open("NetAmount.txt","w+")
    textOrigin.close()
    textOrigin = open("TotalAmount.txt","w+")
    textOrigin.close()
    textOrigin = open("ChineseAmount.txt","w+")
    textOrigin.close()
    textOrigin = open("TaxType.txt","w+")
    textOrigin.close()
    textOrigin = open("Date.txt","w+")
    textOrigin.close()
    textOrigin = open("SellerTaxIDNumber.txt","w+")
    textOrigin.close()
    textOrigin = open("Month.txt","w+")
    textOrigin.close()
    textOrigin = open("Year.txt","w+")
    textOrigin.close()


def OCR(image_dir, dec, rec, rec_dict, output):
    args=utility.parse_args()
    args.use_gpu=False
    args.use_mp=True
    args.enable_mkldnn =True
    args.image_dir= image_dir
    args.det_model_dir= dec
    args.rec_model_dir= rec 
    args.rec_char_dict_path= rec_dict
    main(args, output)    



def uniqueFileName(target, output):
    df = pd.read_csv(target,sep='\t',index_col=0)
    df =df.astype(str)
    kk=df.groupby(['file']).apply(lambda group: '&'.join(group['prediction']))
    kk.to_csv(output, sep='\t', mode='a')




@app.after_request
def apply_caching(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

    
@app.route('/Invoice_Detection')
def index():
    return 'hello!!'

@app.route('/Invoice_Detection/predict', methods=['POST'])

def postInput():
    
    ## make check file
    textOrigin = open("C1.txt","w+")
    textOrigin.close()
    
    ## remove cache to save vRAM    
    torch.cuda.empty_cache()
    gc.collect()

    ##  makeRecordFile to save OCR result
    makeRecordFile()
    
    ##  delete last file
    deleteAll()
    
    ## save upload file
    uploaded_files = request.files.getlist("file")
    for f in uploaded_files:
        upload_path = os.path.join('./yolov5/InputImage',f.filename) 
        f.save(upload_path)
        shutil.copy2(upload_path, './backup')
    
    ## Start yolo
    start_time = time.time()
    
    ## yolo normalization
    opt = parse_opt()
    opt.weights = './yolov5/normalweights/best.pt'
    opt.imgsz = (860, 1280)
    opt.source = './yolov5/InputImage'
    opt.save_txt = True
    opt.conf_thres = 0.75
    run(**vars(opt))
    normalize()
    shutil.rmtree('./runs')
    
    ## yolo identificaton and images classification
    opt2 = parse_opt()
    opt2.weights = './yolov5/noteweights/best.pt'
    opt2.imgsz = (860, 1280)
    opt2.save_txt = True
    opt2.source = './yolov5/Invoice_Normalization/output'
    opt2.conf_thres = 0.25
    run(**vars(opt2))
    detect_export_tagged_image()
    
    ##  remove cache to save vRAM 
    torch.cuda.empty_cache()
    gc.collect()
    
    
    ## Record  yolo_identified_time
    yolo_identified_time =(time.time() - start_time)
    
    
    
    ## Start OCR
    start2_time = time.time()
    OCR("./outputImage/tagged_image/InvoiceNumber/", "./inference/det/", "./inference/IN_SeID_rec/", "./ppocr/utils/ppocr_keys_v1.txt", "./InvoiceNumber.txt")
    OCR("./outputImage/tagged_image/Buyer/", "./inference/det/", "./inference/BY/", "./ppocr/utils/BuyerNam_dict.txt", "./Buyer.txt")
    OCR("./outputImage/tagged_image/BuyerTaxIDNumber/", "./inference/det/", "./inference/BID/", "./ppocr/utils/money_number_dict.txt", "./BuyerTaxIDNumber.txt")
    OCR("./outputImage/tagged_image/Date/", "./inference/det/", "./inference/DANumber/", "./ppocr/utils/ic15_dict.txt", "./Date.txt")
    OCR("./outputImage/tagged_image/NetAmount/", "./inference/det/", "./inference/Money_rec/", "./ppocr/utils/money_number_dict.txt", "./NetAmount.txt")
    OCR("./outputImage/tagged_image/Tax/", "./inference/det/", "./inference/Money_rec/", "./ppocr/utils/money_number_dict.txt", "./Tax.txt")
    OCR("./outputImage/tagged_image/ChineseAmount/", "./inference/det/", "./inference/CA7/", "./ppocr/utils/CA_dict.txt", "./ChineseAmount.txt")
    OCR("./outputImage/tagged_image/TotalAmount/", "./inference/det/", "./inference/Money_rec/", "./ppocr/utils/money_number_dict.txt", "./TotalAmount.txt")
    OCR("./outputImage/tagged_image/Year/", "./inference/det/", "./inference/IN_SeID_rec/", "./ppocr/utils/ppocr_keys_v1.txt", "./Year.txt")
    OCR("./outputImage/tagged_image/Month/", "./inference/det/", "./inference/DANumber/", "./ppocr/utils/ic15_dict.txt", "./Month.txt")
    OCR("./outputImage/tagged_image/SellerTaxIDNumber/", "./inference/det/", "./inference/IN_SeID_rec/", "./ppocr/utils/ppocr_keys_v1.txt", "./SellerTaxIDNumber.txt")
    args=utility.parse_args()
    args.use_gpu=False
    args.image_dir="./outputImage/tagged_image/Note/" 
    args.det_model_dir="./inference/det/"
    args.rec_model_dir="./inference/NOTE_last/"
    args.rec_char_dict_path="./ppocr/utils/ppocr_keys_v1.txt"
    main2(args)
    
    ## Record  yolo_identified_time
    OCR_time = (time.time() - start2_time)
    

    ## turn txt result to json
    uniqueFileName('./InvoiceNumber.txt', './InvoiceNumber2.txt')
    InvoiceNumber = process_text_to_json('./InvoiceNumber2.txt','InvoiceNumber',"InvoiceNumber.json")
    uniqueFileName('./Buyer.txt', './Buyer2.txt')
    Buyer = process_text_to_json('./Buyer2.txt','Buyer','Buyer.json')
    uniqueFileName('./BuyerTaxIDNumber.txt', './BuyerTaxIDNumber2.txt')
    BuyerTaxIDNumber = process_text_to_json('./BuyerTaxIDNumber.txt','BuyerTaxIDNumber','BuyerTaxIDNumber.json')
    uniqueFileName('./Date.txt', './Date2.txt')
    Date = process_text_to_json('./Date2.txt','Date','Date.json')
    uniqueFileName('./NetAmount.txt', './NetAmount2.txt')
    uniqueFileName('./Note2.txt', './Note3.txt')
    Note = process_text_to_json('./Note3.txt','Note','Note.json')
    NetAmount = process_text_to_json('./NetAmount2.txt','NetAmount','NetAmount.json')
    uniqueFileName('./SellerTaxIDNumber.txt', './SellerTaxIDNumber2.txt')
    SellerTaxIDNumber = process_text_to_json('./SellerTaxIDNumber2.txt','SellerTaxIDNumber','SellerTaxIDNumber.json')
    uniqueFileName('./Tax.txt', './Tax2.txt')
    Tax = process_text_to_json('./Tax2.txt','Tax','Tax.json')
    uniqueFileName('./ChineseAmount.txt', './ChineseAmount2.txt')
    ChineseAmount = process_text_to_json_ChineseAmount('./ChineseAmount2.txt','ChineseAmount','ChineseAmount.json')
    uniqueFileName('./TotalAmount.txt', './TotalAmount2.txt')
    TotalAmount = process_text_to_json('./TotalAmount2.txt','TotalAmount','TotalAmount.json')
    uniqueFileName('./Year.txt', './Year2.txt')
    Year = process_text_to_json('./Year2.txt','Year','Year.json')
    uniqueFileName('./Month.txt', './Month2.txt')
    Month = process_text_to_json('./Month2.txt','Month','Month.json')
    taxtypefilenames = listdir('./yolov5/InputImage')
    for f in taxtypefilenames:
        k = str(f).replace('.png','')
        k = k.replace('.jpg','')
        k = k.replace('.JPG','')
        k = k.replace('.PNG','')
        k = k.replace('.bmp','')
        dataJsoos ={
                "name": k,
                'TaxType':'應稅'
            }
        with open('TaxType.json', "a") as outfile:
            json.dump(dataJsoos, outfile)
            outfile.write("\n")
    
    
    
    yolofilenames = listdir('./yolov5/InputImage')
    yolo_identified_time = yolo_identified_time/len(yolofilenames)
    for f in yolofilenames:
        tt = str(f).replace('.png','')
        tt = tt.replace('.jpg','')
        tt = tt.replace('.JPG','')
        tt = tt.replace('.PNG','')
        tt = tt.replace('.bmp','')
        datayolo ={
                "name": tt,
                'yolo_identified_time':"%s s" % yolo_identified_time
            }
        with open('yolo_identified_time.json', "a") as outfile:
            json.dump(datayolo, outfile)
            outfile.write("\n")
            
    OCRfilenames = listdir('./yolov5/InputImage')
    OCR_time = OCR_time/len(OCRfilenames)
    for f in OCRfilenames:
        jj = str(f).replace('.png','')
        jj = jj.replace('.jpg','')
        jj = jj.replace('.JPG','')
        jj = jj.replace('.PNG','')
        jj = jj.replace('.bmp','')
        dataOCR ={
                "name": jj,
                'OCR_time':"%s s" % OCR_time
            }
        with open('OCR_time.json', "a") as outfile:
            json.dump(dataOCR, outfile)
            outfile.write("\n")


    ## process exported json with pandas dataframe
    df1 = pd.read_json('InvoiceNumber.json', lines=True)
    df2 = pd.read_json('Buyer.json', lines=True)
    df3 = pd.read_json('BuyerTaxIDNumber.json', lines=True)
    df4 = pd.read_json('Date.json', lines=True)
    df5 = pd.read_json('Note.json', lines=True)
    df6 = pd.read_json('NetAmount.json', lines=True)
    df7 = pd.read_json('SellerTaxIDNumber.json', lines=True)
    df8 = pd.read_json('Tax.json', lines=True)
    df9 = pd.read_json('TotalAmount.json', lines=True)
    df10 = pd.read_json('Year.json', lines=True)
    df11 = pd.read_json('Month.json', lines=True)
    df12 = pd.read_json('ChineseAmount.json', lines=True)
    df13 = pd.read_json('TaxType.json', lines=True)
    df14 = pd.read_json('yolo_identified_time.json', lines=True)
    df15 = pd.read_json('OCR_time.json', lines=True)
    
    df = df1.merge(df2, on='name')
    df = df.merge(df3, on='name')
    df = df.merge(df4, on='name')
    df = df.merge(df5, on='name')
    df = df.merge(df6, on='name')
    df = df.merge(df7, on='name')
    df = df.merge(df8, on='name')
    df = df.merge(df9, on='name')
    df = df.merge(df10, on='name')
    df = df.merge(df11, on='name')
    df = df.merge(df12, on='name')
    df = df.merge(df13, on='name')
    df = df.merge(df14, on='name')
    df = df.merge(df15, on='name')
    result = df.to_json(orient="split")
    parsed = json.loads(result)
    json.dumps(parsed, indent=4)
    
    ## remove intermediate file
    removefile()    
    
    
    
    
    
    
    ## remove check file
    os.remove("C1.txt")

    return json.dumps(parsed, indent=4,ensure_ascii=False)
                   
                  

if __name__ == '__main__':
    app.run(host='10.192.220.10', port=8689, debug=True,threaded=True)


