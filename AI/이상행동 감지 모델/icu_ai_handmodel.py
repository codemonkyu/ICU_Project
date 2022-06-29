import mediapipe as mp
import cv2
import numpy as np
import natsort
import time
import uuid
import os
import paho.mqtt.publish as publisher
import paho.mqtt.client as mqtt
from threading import Thread

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


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
        try:
            msg = message.payload.decode("utf-8")  #여기가 움직임 센서 감지시 신호가 오는 부분(토픽명:camera/dt, 메세지:"1"로 도착할 예정)
            ######### 사진 경로 ################
            list1 = msg.split(',')
            date = str(list1[0])
            time_ = str(list1[1])
            topic=message.topic
            
            print(topic+":"+msg)

            if(topic=="camera/dt"):

                def recentfile(folder):

                    # 가장 최근에 만들어진 파일 가져오기
                    files_Path = folder + "/"  # 파일들이 들어있는 폴더
                    file_name_and_time_lst = []

                    # 해당 경로에 있는 파일들의 생성시간을 함께 리스트로 넣어줌.
                    for f_name in os.listdir(f"{files_Path}"):
                        written_time = os.path.getctime(f"{files_Path}{f_name}")
                        file_name_and_time_lst.append((f_name, written_time))

                    # 생성시간 역순으로 정렬하고,
                    sorted_file_lst = natsort.natsorted(file_name_and_time_lst, reverse=True)

                    # 가장 앞에 있는 놈을 넣어준다.
                    recent_file = sorted_file_lst[0]
                    recent_file_name = recent_file[0]
                    return (recent_file_name)

                ###############################
                # 키패드 좌표 설정
                left_x_key, left_y_key, right_x_key, right_y_key = 331,715,398,883

                # 문손잡이 좌표 설정
                left_x_handle, left_y_handle, right_x_handle, right_y_handle = 371,892,474,1036
                ##############################

                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_hands = mp.solutions.hands

                pic = ""
                num = 1
                time.sleep(1)

                while True:

                    # For static images:
                    num = num + 2
                    # recentfilename = recentfile('/home/lab45/cctv/images/' + date + "/" + time_)

                    recentfilename = f'{num:04}' + ".jpg"

                    try:
                        IMAGE_FILES = ['/home/lab45/cctv/images/' + date + '/' + time_ + '/' + recentfilename]
                        print(recentfilename)
                    #
                    # if recentfilename == pic:
                    #     break

                    # 모델에 들어간 사진 이름 저장
                    # pic = recentfilename[:]

                        with mp_hands.Hands(
                                static_image_mode=True,
                                max_num_hands=2,
                                min_detection_confidence=0.5) as hands:

                            ex = ''
                            for idx, file in enumerate(IMAGE_FILES):

                                # Read an image, flip it around y-axis for correct handedness output (see
                                # above).
                                image = cv2.flip(cv2.imread(file), 1)

                                # Convert the BGR image to RGB before processing.
                                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                                if results.multi_hand_landmarks:
                                    for handLms in results.multi_hand_landmarks:
                                        for id, lm in enumerate(handLms.landmark):
                                            h, w, c = image.shape

                                if not results.multi_hand_landmarks:
                                    continue

                                image_height, image_width, _ = image.shape
                                annotated_image = image.copy()

                                for hand_landmarks in results.multi_hand_landmarks:
                                    mp_drawing.draw_landmarks(
                                        annotated_image,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())

                                ##### 사진 저장 경로 설정 ###########
                                cv2.imwrite('/home/lab45/cctv/ai_images/' + str(idx) + '.png', cv2.flip(annotated_image, 1))

                                try:
                                    finger_one = [results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                                  results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP].y]
                                    one_x, one_y = int(finger_one[0] * w), int(finger_one[1] * h)

                                    finger_two = [
                                        results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                        results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y]
                                    two_x, two_y = int(finger_two[0] * w), int(finger_two[1] * h)

                                    finger_three = [
                                        results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                        results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y]
                                    three_x, three_y = int(finger_three[0] * w), int(finger_three[1] * h)

                                    finger_one_x = w - one_x
                                    finger_two_x = w - two_x
                                    finger_three_x = w - three_x

                                    if ((left_x_key <= finger_two_x) & (finger_two_x <= right_x_key)) | (
                                            (left_x_key <= finger_one_x) & (finger_one_x <= right_x_key)) | (
                                            (left_x_key <= finger_three_x) & (finger_three_x <= right_x_key)):
                                        if ((left_y_key <= two_y) & (two_y <= right_y_key)) | (
                                                (left_y_key <= one_y) & (one_y <= right_y_key)) | (
                                                (left_y_key <= three_y) & (three_y <= right_y_key)):
                                            print('1')
                                            ex = 'exit'
                                            publisher.single("camera/ai", "1", hostname="52.79.159.146")
                                            break;

                                    if ((left_x_handle <= finger_two_x) & (finger_two_x <= right_x_handle)) | (
                                            (left_x_handle <= finger_one_x) & (finger_one_x <= right_x_handle)) | (
                                            (left_x_handle <= finger_three_x) & (finger_three_x <= right_x_handle)):
                                        if ((left_y_handle <= two_y) & (two_y <= right_y_handle)) | (
                                                (left_y_handle <= one_y) & (one_y <= right_y_handle)) | (
                                                (left_y_handle <= three_y) & (three_y <= right_y_handle)):
                                            print('1')
                                            ex = 'exit'
                                            publisher.single("camera/ai", "1", hostname="52.79.159.146")
                                            break;

                                    else:
                                        print('0')

                                except:
                                    pass

                            if ex=='exit':
                                break
                    except:
                        break



        except Exception as e:
            print(e)

        finally:
            pass
   
if __name__ =="__main__":
    mymqtt= MqttWorker()
    mymqtt.mymqtt_connect()
