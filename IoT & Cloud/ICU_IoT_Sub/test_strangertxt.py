import os
file = "/home/ubuntu/cctv/images/2022년06월23일/22시43분/"+"stranger.txt"
if os.path.isfile(file):
    print("stranger.txt 파일이 존재합니다.")
    
else:
    print("없습니다.")