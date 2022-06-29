from distutils.log import error
import paho.mqtt.client as mqtt
from threading import Thread
import paho.mqtt.publish as publisher
import base64
import os
import filefunc as filefunc
import shutil
import datetime

class MqttWorker:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.frame = None

        
    def mymqtt_connect(self): # 사용자정의 함수 - mqtt서버연결과 쓰레드생성 및 시작을 사용자정의 함수로 정의
        try:
            print("영상저장 브로커 연결 시작")
            self.client.connect("52.79.159.146",1883,60)
            mythreadobj = Thread(target=self.client.loop_forever)
            mythreadobj.start()
            
        except KeyboardInterrupt:
            pass
            
        finally:
            pass 
 
               
    def on_connect(self,client, userdata, flags, rc): # broker접속에 성공하면 자동으로 호출되는 callback함수
        print("connect..."+str(rc)) # rc가 0이면 성공 접속, 1이면 실패
        if rc==0 : #연결이 성공하면 구독신청
            client.subscribe("camera/#")
            client.subscribe("listplz")

        else:
            print("연결실패.....")   
            
    def on_disconnect(self, client, userdata, rc):
        if rc == 0:
            #self.mqtt_connect = False
            print("MQTT 브로커 연결 끊김")
            self.disconnected()
        
                
    # 라즈베리파이가 메시지를 받으면 호출되는 함수이므로 받은 메시지에 대한 처리를 구현
    def on_message(self,client, userdata, message): 
        try:
            date=filefunc.recentfile('/home/ubuntu/cctv/images')
            time=filefunc.recentfile('/home/ubuntu/cctv/images/'+ date)
            
            
            if message.topic=="camera/dt":#detected된 날짜 시간을 이름으로 하는 폴더 생성
                msg= message.payload.decode('utf-8')
               
                list1=msg.split(",")
               
                fdate = str(list1[0])
               
                ftime = str(list1[1])
                print(fdate+ftime)
                if not os.path.exists("/home/ubuntu/cctv/images/"+fdate): 
                    os.mkdir("/home/ubuntu/cctv/images/"+fdate)
                if not os.path.exists("/home/ubuntu/cctv/images/"+fdate+"/"+ftime): 
                    os.mkdir("/home/ubuntu/cctv/images/"+fdate+"/"+ftime)

            elif message.topic=="camera/frame":  #프레임 전송 받기(jpg파일로 저장)
                
                date=filefunc.recentfile('/home/ubuntu/cctv/images')
                time=filefunc.recentfile('/home/ubuntu/cctv/images/'+ date)
                
                order = "ffmpeg -f image2 -r 10 -i /home/ubuntu/cctv/images/"+date+"/"+time+"/%04d.jpg -c:v libx264 output.mp4"
                msg = message.payload.decode("utf-8")
                if msg=="complete":
                    os.system(order)
                    print("mp4변환 완료")
                    file_source = 'output.mp4' #output.mp4 저장된 위치
                    file_destination = '/home/ubuntu/cctv/static/recordvideo/' #영상들 저장할 폴더(버킷)
                    shutil.move(file_source, file_destination + date+time+".mp4")
                    
                    #stranger.txt파일이 존재하면 냅두고 존재 안하면 영상과 폴더 삭제
                    file = "/home/ubuntu/cctv/images/"+date+"/"+time+"/stranger.txt"
                    if os.path.isfile(file):
                        print("stranger.txt 파일이 존재합니다.")
                    else:
                        #영상 삭제
                        os.remove("/home/ubuntu/cctv/static/recordvideo/"+date+time+".mp4")
                        #폴더 삭제
                        dir_path ="/home/ubuntu/cctv/images/"+date+"/"+time
                        shutil.rmtree(dir_path)
                        
                            
                    
                else:
                #가장 최근에 만들어진 date폴더 내의 time폴더
                    flist=msg.split("^time^")
                    filename = flist[0] #txt파일 이름(시+분+초+밀리초)
                    frame = flist[1] #프레임(저장될 내용)
                    #.jpg로 변환
                    frame=frame.encode("utf-8")
                    frame=base64.b64decode(frame)
                    f=open('/home/ubuntu/cctv/images/'+date+"/"+time+"/"+filename+'.jpg',"wb")
                    f.write(frame)
                    f.close()
                    print(".jpg저장완료") 
                
            elif message.topic == "camera/ai":
                print("AI: 이상행동 발생!!")   
                
                #if : stranger있는지 확인! 없으면 만들고 신호보내/ 있으면 신호 안보내
                result = os.path.isfile("/home/ubuntu/cctv/images/"+date+"/"+time+"/stranger.txt")
                if result == False:
                    f=open("/home/ubuntu/cctv/images/"+date+"/"+time+"/stranger.txt","a")
                    f.write(datetime.datetime.now().strftime("%H시%M분%S초%f")) 
                    f.close() 
                    #신호보내
                    publisher.single("signal","1",hostname="52.79.159.146")
                else:
                    print("strange.txt파일이 존재")
                
            elif message.topic == "listplz":
                file_list = os.listdir("/home/ubuntu/cctv/static/recordvideo")
                # file_list = os.listdir("C:/2022_IoT/ICU/static")
                mylist=""
                for i in file_list:
                    mylist =mylist+i+","
                publisher.single("recordfolder",mylist,hostname="52.79.159.146")
                print("폴더 리스트 전달 완료")        

        except Exception as e:
            print(e)
        finally:
            pass

    
            
if __name__ =="__main__":
    mymqtt= MqttWorker()
    mymqtt.mymqtt_connect()    
