import paho.mqtt.client as mqtt
from threading import Thread
import argparse
import os
import natsort
import time
import paho.mqtt.publish as publisher
import sys
import math
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

ROOT = 'home/lab45/yolov5'  # YOLOv5 root directory

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# @torch.no_grad()

# def recentfile(folder):
#
#     #가장 최근에 만들어진 파일 가져오기
#     files_Path = folder+"/" # 파일들이 들어있는 폴더
#     file_name_and_time_lst = []
#
#     # 해당 경로에 있는 파일들의 생성시간을 함께 리스트로 넣어줌.
#     for f_name in os.listdir(f"{files_Path}"):
#         written_time = os.path.getctime(f"{files_Path}{f_name}")
#         file_name_and_time_lst.append((f_name, written_time))
#
#     # 생성시간 역순으로 정렬하고,
#     sorted_file_lst = natsort.natsorted(file_name_and_time_lst, reverse=True)
#
#     # 가장 앞에 있는 놈을 넣어준다.
#     recent_file = sorted_file_lst[0]
#     recent_file_name = recent_file[0]
#     return(recent_file_name)

device=''
weights='/home/lab45/yolov5/runs/train/yolo_0621_result/weights/best2.pt'
dnn=False
half=False
imgsz=(640, 640)

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

class MqttWorker:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
     
    def mymqtt_connect(self): # 사용자정의 함수 - mqtt서버연결과 쓰레드생성 및 시작을 사용자정의 함수로 정의
        try:
            print("브로커 연결 시작하기")
            self.client.connect("52.79.159.146",1883,60) 
            mythreadobj = Thread(target=self.client.loop_forever)
            mythreadobj.start()          
        except KeyboardInterrupt:
            pass       
        finally:
            print("종료~~~~") 
 
    def on_connect(self,client, userdata, flags, rc): # broker접속에 성공하면 자동으로 호출되는 callback함수
        print("connect..."+str(rc)) # rc가 0이면 성공 접속, 1이면 실패
        if rc==0 : #연결이 성공하면 구독신청
            client.subscribe("camera/dt") #토픽명: camera/dt인 신호를 받겠다.
        else:
            print("연결실패.....")   
            
    def on_disconnect(self, client, userdata, rc):
        if rc == 0:
            print("MQTT 브로커 연결 끊김")
            self.disconnected()
        
                
    # 라즈베리파이가 메시지를 받으면 호출되는 함수이므로 받은 메시지에 대한 처리를 구현
    def on_message(self,client, userdata, message):
        visualize=False
        project= './runs/detect'
        name='exp'
        exist_ok=False
        line_thickness=3
        hide_labels=False
        augment=False
        conf_thres=0.25
        iou_thres=0.45
        classes=None
        agnostic_nms=False
        max_det=1000
        save_crop=False
        save_conf=False
        hide_conf=False
        save_txt=True
        now = ''
        try:
            msg = message.payload.decode("utf-8")  #여기가 움직임 센서 감지시 신호가 오는 부분(토픽명:camera/dt, 메세지:"1"로 도착할 예정)
            list1=msg.split(",")
            date = str(list1[0])
            time_ = str(list1[1])
            topic=message.topic

            print(topic+":"+msg)

            if(topic=="camera/dt"):
                print("AI MODEL LOAD") #여기에 AI 모델 Load되는 코드가 들어가면 됩니다.

                num = 1
                time.sleep(5)

                while True:
                    num = num + 2

                    recentfilename = f'{num:04}' + ".jpg"

                    try:
                        img_path = '/home/lab45/cctv/images/'+ date + '/'+ time_ + '/' + recentfilename

                        # detect.py
                        print(recentfilename)
                        source = str(img_path)

                        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)

                        # Dataloader
                        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
                        bs = 1  # batch_size
                        # vid_path, vid_writer = [None] * bs, [None] * bs

                        # Run inference
                        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
                        dt, seen = [0.0, 0.0, 0.0], 0
                        for path, im, im0s, vid_cap, s in dataset:
                            t1 = time_sync()
                            im = torch.from_numpy(im).to(device)
                            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                            im /= 255  # 0 - 255 to 0.0 - 1.0
                            if len(im.shape) == 3:
                                im = im[None]  # expand for batch dim
                            t2 = time_sync()
                            dt[0] += t2 - t1

                            # Inference
                            visualize = increment_path(save_dir / Path(path).stem, mkdir=False) if visualize else False
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

                                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                                p = Path(p)  # to Path

                                save_path = str(save_dir / p.name)  # im.jpg
                                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                                s += '%gx%g ' % im.shape[2:]  # print string
                                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                imc = im0.copy() if save_crop else im0  # for save_crop
                                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                                # 중심좌표 변수
                                hand = []
                                key = []
                                if len(det):
                                    # Rescale boxes from img_size to im0 size
                                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                                    # Print results
                                    for c in det[:, -1].unique():
                                        n = (det[:, -1] == c).sum()  # detections per class
                                        s += f"a{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                                    # Write results
                                    for *xyxy, conf, cls in reversed(det):
                                        if save_txt:  # Write to file
                                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                            # with open(f'{txt_path}.txt', 'a') as f:
                                            #     f.write(('%g ' * len(line)).rstrip() % line + 'b\n')

                                            c = int(cls)  # integer class

                                            # kkk = xyxy[0].eval(session=tf.Session())

                                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')#여기가 이름
                                            annotator.box_label(xyxy, f'{label}', color=colors(c, True))

                                            #중심 좌표 저장
                                            if names[c] == 'key':
                                              key.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                                            elif names[c] == 'hand':
                                              hand.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                                        if save_crop:
                                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'd{p.stem}.jpg', BGR=True)

                                #한프레임 후처리
                                grab = 0
                                if (key != None) and (hand != None):
                                  for k in key:
                                    for h in hand:
                                      k_midd = [(k[0] + k[2])/2, (k[1]+k[3])/2]
                                      k_hor = (k[2] - k[0])/2

                                      h_midd = [(h[0] + h[2])/2, (h[1]+h[3])/2]
                                      h_hor = (h[2] - h[0])/2

                                      c = math.sqrt( (k_midd[0]-h_midd[0])**2 + (k_midd[1]-h_midd[1])**2 )

                                      if c <= (k_hor + h_hor):
                                        grab = grab + 1

                                #최종 출력
                                if grab >0:
                                  r = 1

                                else:
                                  r = 0
                                annotator.box_label([0,0,0,0], f'{r}' , color=colors(3, True)) #소히

                            print(r)

                            if r==1:
                              publisher.single("camera/ai", "1", hostname="52.79.159.146")
                              break

                        if r==1:
                          break

                    except :
                        break

        finally:
            pass
   
if __name__ =="__main__":
    mymqtt= MqttWorker()
    mymqtt.mymqtt_connect()    
