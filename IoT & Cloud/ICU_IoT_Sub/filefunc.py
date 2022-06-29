import os
from datetime import datetime


def whatfile(folder,type):
    dir_path = folder
    file_list=os.listdir(dir_path)
    for i in range(0,len(file_list),1):
        if type not in file_list[i]:
            file_list.remove(file_list[i])
    file_list.sort()
    return file_list

def makedir():
    try:#record안에 날짜를 이름으로 하는 파일 생성
        if not os.path.exists("/home/ubuntu/video/camera/"+datetime.today().strftime("%Y%m%d")):
            os.mkdir("/home/ubuntu/video/camera/"+datetime.today().strftime("%Y%m%d")) 
        f=open("/home/ubuntu/video/camera/"+datetime.today().strftime("%Y%m%d")+"/"+datetime.today().strftime("%H%M")+".txt",'w')
        f.close()
    except OSError:
        pass  
    
def makedir2(fdate,ftime):
    if not os.path.exists("/home/ubuntu/cctv/images/"+fdate): 
        os.mkdir("/home/ubuntu/cctv/images/"+fdate)
    if not os.path.exists("/home/ubuntu/cctv/images/"+fdate+"/"+ftime): 
        os.mkdir("/home/ubuntu/cctv/images/"+fdate+"/"+ftime)
    
def recentfile(folder):
        #가장 최근에 만들어진 파일 가져오기
    files_Path = folder+"/" # 파일들이 들어있는 폴더
    file_name_and_time_lst = []
    # 해당 경로에 있는 파일들의 생성시간을 함께 리스트로 넣어줌. 
    for f_name in os.listdir(f"{files_Path}"):
        written_time = os.path.getctime(f"{files_Path}{f_name}")
        file_name_and_time_lst.append((f_name, written_time))
    # 생성시간 역순으로 정렬하고, 
    sorted_file_lst = sorted(file_name_and_time_lst, key=lambda x: x[1], reverse=True)
    # 가장 앞에 이는 놈을 넣어준다.
    recent_file = sorted_file_lst[0]
    recent_file_name = recent_file[0]
    return(recent_file_name)

def whatfolder(basepath):#폴더명 안에 있는 폴더이름 가져와서 111,222,333 이런식으로 문자열로 출력
    folderlist=[]
    result=""
    with os.scandir(basepath) as entries:
        for entry in entries:
            if entry.is_dir():
                folderlist.append(entry.name)

    for i in range(len(folderlist)):
        result = result + folderlist[i] +","
    result= result[:-1]
    return(result)

def dtmake(msg):
    list1=msg.split(",")
    fdate = str(list1[0])
    ftime = str(list1[1])
    print(fdate+ftime)
    makedir2(fdate,ftime) 
    #이상행동이 감지되었다고 가정하고 임시로 stranger.txt파일을 생성해둠(이거 생성안해두면 자동 삭제되게함)       
    f=open("/home/ubuntu/cctv/images/"+fdate+"/"+ftime+"/stranger.txt","a")
    f.write(fdate+ftime) 
    f.close()