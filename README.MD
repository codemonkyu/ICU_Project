## 융복합 프로젝트 클라우드 부분 기능 구현 및 트러블 슈팅 정리

### Ubuntu에 mqtt서버(broker) 설치하기

---

##### 설치환경

Ubuntu 20.04 LTS

##### 라이브러리 설치

$ sudo apt-get update

$ sudo apt-get install build-essential libc-ares-dev uuid-dev libssl-dev

$ libcurl4-openssl-dev libmysqlclient-dev

##### 폴더 만들고 1.4.10 버전 다운받아서 압축 풀기

$ cd ~/

$ mkdir mosquitto_sources

$ cd mosquitto_sources/

$ wget http://mosquitto.org/files/source/mosquitto-1.4.10.tar.gz

$ tar xvf mosquitto-1.4.10.tar.gz

##### libwebsockets 설치 (라이브러리)

$ sudo apt-get install libwebsockets-dev

##### config.mk 파일 수정

$ WITH_WEBSOCKETS:=yes 로 변경 후 저장

##### build해서 설치하기

$ cd ~/mosquitto_sources/mosquitto-1.4.10/

$ make
$ sudo make install

##### mosquitto.conf 파일에서 아래 내용 추가하기

listener 9001
protocol websockets

listener 1883
protocol mqtt

##### Start Broker / 수정한 conf 파일로 시작하기

$ mosquitto -c mosquitto.conf -v

#### 지켜야할 사항

1. 꼭 위의 순서를 지켜서 설치해주자 특히 make 전에 conf파일을 수정해줘야하고 이후 make install 해줘야한다.

<HR></HR>

### IoT에서 수신하는 mqtt데이터를 jpg파일로 변환 후 보관할 DB 구축

#### goofys를 사용하면 쉽게 s3버킷을 EC2에 마운트 해줄 수 있다.

1. S3 만드는 계정의 액세스키 발급받기
2. Amazone s3 콘솔에서 s3 버킷 default 값으로 생성 / 리전확인!!
3. ec2 접속
4. Ubuntu 에서 s3 버컷 마운트해주기

$ mkdir [마운트할 경로]

$ apt-get update

$ apt-get install -y fuse awscli

$ aws configure →
access key /
secret access key /
region : eu-west-2 /
data : json

$ wget https://github.com/kahing/goofys/releases/latest/download/goofys

$ chmod +x goofys

$ /goofys [s3버킷 이름] [마운트할 경로]

- 제대로마운트 해줬다면 EC2의 폴더에서 S3 버킷의 객체를 확인 할 수 있다.
