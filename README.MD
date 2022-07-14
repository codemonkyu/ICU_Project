## 융복합 프로젝트 (AI, Big data, IoT, Cloud) _ 7조 _ ICU \_ 포트폴리오

---

#### 🖋SUMMARY

- 프로젝트 기간
  - 5월 23일 ~ 6월 28일

</br>

- 프로젝트 주제
  - 스토킹으로 인한 2차 범죄 예방을 위한 지능형 CCTV 개발

</br>

- 프로젝트 기획의도

  - 증가하는 스토킹
  - 피해자 보호 미흡
  - 단순 스토킹에서 흉악범죄로 이어지는 스토킹
  - CCTV의 범죄 예방 효과를 이용한 스토킹으로 인한 범죄 예방안 제시

</br>

- 프로젝트 목표
  - 지능형 CCTV 설치를 통한 개인 보안 강화 (실시간 영상 확인)
  - 스토킹 영상의 저장을 통한 증거 자료 수집 (영상 저장)
  - 이상행동 감지를 통한 스토킹 이후 2차 범죄를 예방 (이상행동 감지 및 알림)
  - 추후 효율적인 서비스 확대를 위한 지능형 CCTV의 우선입지 선정 조사

</br>

- 💬 지능형 cctv의 기능 구현 목표:

<B>💾 1. 증거자료 확보:</B> 사람의 이상행동을 감지하는 지능형 CCTV가 이상행동(스토킹과 비슷한 상황)을 스스로 감지하고 영상을 저장하여 범죄에 대한 증거자료를 확보함.

<B>📩 2. 실시간 영상 전송 및 알림:</B> 이상행동을 감지한 지능형 CCTV가 사용자의 휴대폰에 실시간으로 알림을 전달, 알림을 확인한 사용자는 실시간 영상 및 저장된 영상을 확인하여 상황을 파악함.

<B>☎ 3. 응급 전화 및 SMS 전송 기능:</B> 실시간 알림확인 후 상황을 파악한 사용자가 신속하게 위험상황에 대처가능함.

---

#### 👨‍👨‍👧‍👦 분야별 팀원 소개

- AI, Big data, IoT, Cloud 분야별 팀원 소개

#### 📖 프로젝트 구성도 소개

- 서비스 전체 구성도 및 상세 구성도

#### 📺 프로젝트 기능 소개 (시연영상, PPT)

- 서비스 전체 시연영상 및 PPT

#### 📚 회의록

- Notion 칸반 보드를 사용한 협업 및 주차별 수행일지
- 전체 WBS

#### 😀 느낀점

- AI, BIG DATA, IoT등 다른 IT 분야에 대해 공부해본적이 없어 이해도가 전혀 없었던 상태에서 융복합 프로젝트를 진행하면서 어려움이 많았습니다. 하지만 클라우드에서 모든 분야의 기술을 유기적으로 연결시키고 작동하도록 해야 했습니다. 때문에 필수적으로 다른 분야를 열심히 공부해야 했습니다. 결과적으로 이번 프로젝트를 통해 클라우드 외의 분야를 이해하고 성공적으로 서비스의 아키텍처를 구성할 수 있었습니다. 또한 제가 평소 사용하던 제품의 서비스가 어떻게 작동하고 구성되어있는지 파악하고 이해할 수 있는 기회였습니다.

<br>

<B>💡 [자세한 정보 보러 가기](https://rustic-mailman-444.notion.site/7-f448609b0c2e4a69b5191be6b305f322) 💡</B>

<br>

---

## [클라우드] 기능 구현 및 트러블 슈팅 정리

---

### 1. Ubuntu에 mqtt서버(broker) 설치하기

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

### 2. IoT에서 수신하는 mqtt데이터를 jpg파일로 변환 후 보관할 S3버킷 구축

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

---

## 3. AWS Cognito / AWS Amplify

AWS Cognito
Amazon Cognito는 웹 및 모바일 앱에 대한 인증, 권한 부여 및 사용자 관리를 제공합니다. 사용자는 사용자 이름과 암호를 사용하여 직접 로그인하거나 Facebook, Amazon, Google 또는 Apple 같은 타사를 통해 로그인할 수 있습니다.

Amazon Cognito의 두 가지 주요 구성 요소는 사용자 풀과 자격 증명 풀입니다. 사용자 풀은 앱 사용자의 가입 및 로그인 옵션을 제공하는 사용자 디렉터리입니다. 자격 증명 풀을 통해 기타 AWS 서비스에 대한 사용자 액세스 권한을 부여할 수 있습니다. 자격 증명 풀과 사용자 풀을 별도로 또는 함께 사용할 수 있습니다.

1. 정리하자면 AWS Cogito는 웹이나 앱에서 사용자 인증처리를 해주는 서비스이다. 사용자 풀 관리를 통해 원하는 속성을 부여해서 어떤 정보를 통해 사용자 정보를 관리하고 인증 처리를 진행할 것인지 설정할 수 있다.
   (예- 최종 사용자 로그인방법, 회원가입시 입력해야할 표준속성필수사항, 비밀번호의 수준)

2. 원하는 속성/정책 등을 설정하여 사용자 풀을 생성했다면 '앱 클라이언트' 부분에서 어떤 서비스에 생성한 사용자 풀에 대한 액세스 권한을 부여할 것인지 설정해줘야한다.

3. 2번에서 생성한 사용자 풀과 앱 클라이언트를 연결해줬다면 인증기능을 구현할 수 있다.
   이때 유저풀의 ID값, 앱클라이언트의 값, 클라이언트 보안키 값 등이 필요하다.

AWS Amplify
AWS Amplify는 사용 사례가 발전함에 따라 프런트엔드 웹 및 모바일 개발자가 다양한 AWS 서비스를 활용하는 유연성을 바탕으로 AWS에 풀 스택 애플리케이션을 쉽고 빠르게 구축할 수 있도록 지원하는 특별히 제작된 도구와 기능 집합입니다. Amplify를 사용하면 몇 분 만에 웹 또는 모바일 앱 백엔드를 구성하고 앱을 연결하며, 웹 프런트엔드 UI를 시각적으로 구축하고, AWS 콘솔 외부 앱 콘텐츠를 쉽게 관리할 수 있습니다. 클라우드 전문 지식이 필요하지 않으며 더 빠르게 배포하고 손쉽게 확장할 수 있습니다.

Android - Amplify 연결

1. 새로운 안드로이드 프로젝트 만들기
2. AWS AppSync SDK 가져오기 및 앱 구성

- build.gradle에 속성 추가

3. AWS Amplify CLI 설치 및 새로운 AWS Amplify 프로젝트 초기화
4. AWS Amplify CLI 설치
5. AWS Amplify 프로젝트 초기화

- amplify init

[출처: aws doc](https://aws.amazon.com/ko/blogs/korea/building-an-android-app-with-aws-amplify-part-1/)

---

## 4. 쉘 스크립트로 오류 찾아 해결하기

AWS RDS를 사용하여 Mysql을 만들어 서비스에서 사용자의 사용기록을 관리하는 DB를 만들었다.
이후 Django의 디폴트 db를 RDS로 만든 Mysql의 엔드포인트를 사용하여 연결해주었다. 이후 docker를 이용해 서비스를 컨테이너화 시켜 배포해주었다.
이후 사용자가 문을 열거나 카메라에 찍히면 IoT의 마그네틱 /움직임 감지 센서에서 mqtt통신을 통해 서비스로 신호를 보내고 서비스는 받은 통신을 기록한다. 이때 기록은 (문열림/움직임의 여부, 시간 등이다.) 하지만 문제점은 여기서 발생했다!!!!

- 문제점1 : 사용자의 서비스 기록 (로그기록)이 하나만 쌓이고 이후 쌓이지 않음!
  <br>
- 문제점2 : 첫번째 쌓인 기록의 시간이 맞지 않는다!
  <br>
- 문제점 1의 해결 방법: mysql workbench에 들어가 DB의 스키마의 설정값을 확인했다. 자동증가 (auto increment ) 가 체크되어 있어야 했는데 체크가 안되어 있었다. 또한 이전에 값이 들어간 상태에서 설정을 변경하는것이 안되어서 모든 db값을 지우고 다시 스키마의 설정을 변경해주었다.
  <br>

* 문제점 2의 해결 방법: django 프레임워크를 사용하여 서비스의 백엔드 쪽을 만들었기 때문에 django의 settings.py에서 time쪽의 시간을 Asia/Seoul로 변경해주어서 해결이 가능했다.
  <br> \*배운점: 이 에러를통해 django 시간설정 / Mysql workbench 사용법을 알게되었다. RDS는 DB는 한번 테이블을 만들고나면 이후 스키마를 수정하는 과정이 쉽지 않다. 하지만 workbench를 사용하면 좀더 수월하게 이런 과정이 가능하다. workbench에 접속하기위해 aws rds를 만들때 퍼블릭 엑세스 가능을 체크해줘야한다.

![image](https://user-images.githubusercontent.com/98939242/178987116-58a60e1a-c233-4e47-8207-69ec5bb696f7.png)

사용자의 사용 로그가 쌓이는 서버 서비스에서의 오류를 잡기위해, 컨테이너화 하여 올린 서비스에 shell로 진입 한다.
(docker exec -it [컨테이너 id] /bin/bash)

1. shell 스크립트 진입 -> python manage.py shell
2. from 앱.models import db명
3. db명 objects.all() 로 db의 저장된 객체 조회
4. i = datetime.datetime.now() 로 i에 현재시간을 변수로 정해준다
5. 이후 i를 입력하여 올바르게 현재시간이 저장되는지 확인한다.

---

## 5. AWS RDS - Mysql 생성하기

- AWS RDS : MySQL DB인스턴스 만들기

1. 리전 선택하기!
2. Create database
3. Select engine에서 Mysql 선택
   (프리티어 사용자일 경우 밑의 only enable options eligible for RDS Free Usage Tier 선택!)
4. DB인스턴스를 구성한다.

- License model: MySQL의 일반 라이선스 계약을 사용하도록 기본값인 general-public-license를 선택합니다. MySQL에는 단 하나의 라이선스 모델만 있습니다.

DB engine version: MySQL의 기본 버전을 선택합니다. 일부 리전에서는 Amazon RDS가 여러 버전의 MySQL을 지원합니다.

DB instance class: db.t2.micro --- 1vCPU, 1 GIB RAM을 선택합니다. 이는 1GB 메모리 및 1vCPU에 해당합니다. 지원되는 인스턴스 클래스 목록을 보려면 Amazon RDS 제품 세부 정보를 참조하십시오.

Multi-AZ deployment: 다중 AZ 배포에 대해서는 비용이 부과됩니다. 다중 AZ 배포를 사용하면 다른 가용 영역에 동기식 예비 복제본을 자동으로 프로비저닝하고 유지합니다.

Storage type: General Purpose (SSD)를 선택합니다. 스토리지에 대한 자세한 내용은 Amazon RDS용 스토리지를 참조하십시오.

Allocated storage: 기본값인 20을 선택하여 데이터베이스에 20GB의 스토리지를 할당합니다. Amazon RDS for MySQL에서는 최대 16TB까지 확장할 수 있습니다.

Enable storage autoscaling: 워크로드가 주기적이거나 예측할 수 없는 경우 스토리지 autoscaling을 활성화하여 필요할 때 RDS가 스토리지를 자동으로 확장하도록 하십시오. 이 자습서에서는 이 옵션을 적용하지 않습니다.
Settings:

DB instance identifier: 선택한 리전의 계정에 대해 고유한 DB 인스턴스 이름을 입력합니다. 본 자습서에서는 이름을 rds-mysql-10minTutorial로 지정합니다.
Master username: DB 인스턴스에 로그인할 때 사용할 사용자 이름을 입력합니다. 본 예제에서는 masterUsername을 사용합니다.
Master password: 마스터 사용자 암호에 8~41개의 인쇄용 ASCII 문자(/, " 및 @ 제외)가 포함된 암호를 입력합니다.

Confirm password: 암호를 다시 입력합니다.

5. Configure Advanced Settings 페이지

- 필요한 추가 정보를 설정을 할 수 있다.

---

## 6. ubuntu 사용자 변경 및 권한 주기

- 리눅스는 멀티유저 운영체재로서 여러명의 사용자가 시스템에 접근이 가능하고 사용자별로 파일및 디렉터리에 대한 소유권/권한을 설정하여 보안적인 부분을 컨트롤 할 수 있다.

* 트러블슈팅 배경: 이번 프로젝트에서 여러 사용자중 하나인 사용자에 특정 디렉터리에 대한 권한을 설정할 일이 생겼다. 자주 사용하는 작업이기 때문에 정리할 필요성을 느꼈다.

- 정리

1. chmod: 파일과 디렉터리의 권한을 변경할 때 사용한다.
   ls -l 명령어를 사용하여 현재 경로에 위차하는 파일들의 권한을 조회해 볼 수 있다.

![image](https://user-images.githubusercontent.com/98939242/178762909-165f4c83-b2ae-4c3b-8ce7-6e8232198b9b.png)

위와 같이 조회 되는 것을 확인 할 수 있다.

여기서 소유자, 소유그룹에 대한 개념을 먼저 숙지하고 있어야한다.
소유자: 파일의 주인을 말한다.
소유그룹: 계정을 그룹화 해놓은 것을 말한다. 즉 소유자가 아니더라도 특정 그룹에 대한 권한을 전체적으로 수정함으로써 그룹 권한으로 파일을 다룰 수 있다.

![image](https://user-images.githubusercontent.com/98939242/178763827-2c760454-763f-447f-bed8-9c455b192a58.png)

위의 사진을 참고하면 먼저 ls -l을 통해 파을들의 권한을 조회한 결과를 이해할 수 있다.

0번 영역: "d"는 디렉토리를 의미한다. 파일은 "-"로 표시가 된다.

123번 영역: 소유자의 사용권한을 의미한다.

456번 영역: 소유 그룹의 사용권한을 의미한다.

789번 영역: 제3자의 사용권한을 의미한다. 제3자는 소유 그룹과 마찬가지로 읽기/실행은 가능하지만 쓰기는 불가능하다.

- r 권한

  - r은 읽는 권한을 의미한다.

- w 권한

  - w권한은 쓰기 권한을 의미한다.

- x 권한

  - x권한은 실행권한을 의미한다.
    -> 대부분의 파일은 쓰기 권한만 있으면 사용가능하다. 하지만 실행파일일 경우 따로 x 권한을 요구한다.

<br>

- 권한 변경하기 (기본)
  - 권한을 바꾸는 명령어는 chmod 이다.

u: 소유자
g: 소유그룹
o: 제3자

+r or -r : 읽기권한 추가 또는 삭제
+w or -w : 쓰기권한 추가 또는 삭제
+x or -x : 실행권한 추가 또는 삭제

ex1) $chmod go+rwx Music
-> music 디렉토리의 소유그룹과 제3자에게 읽기쓰기실행권한을 모두 부여

ex2) $chmod o-xw Music
-> music디렉토리의 제3자에게 실행과 쓰기 권한을 주지 않겠다는 의미이다.

- 권한 변경하기 (숫자를 이용하여 변경하는 방법)

r = 4,
w = 2,
x = 1

ex1) $ chmod 755 Music
총 세 자리의 숫자가 있는데 첫번째 숫자값은 소유자의 권한을 두번째는 소유그룹 세번째는 제3자의 권한에 대한 숫자이다 .

7 = r+w+x 로 읽기 쓰기 실행 권한 모두를 부여한다는 뜻이다.

5 = r+x 로 읽기 실행권한만 모두를 부여한다는 뜻이다.

---

## 7. flask web (python) 웹 배포 정리

-> 서비스의 기획 배경을 설명하고 이후 확장 계획에 대한 근거를 설명해주는 웹페이지를 정적페이지로 만들었다. 정적페이지를 배포해주는 방법은 많았지만, Flask를 통해 만든 정적웹을 EC2를 활용하여 배포해보고싶었다.

1. EC2 생성 (보안그룹에서 TCP 8080 및 80 포트 인바운드 규칙에서 추가)
2. SHH접속으로 EC2 접속하여 만들어 놓은 WEB을 git 등을 이용하여 EC2에 다운받는다.
3. 이후 웹을 실행할때 필요한 라이브러리를 설치해준다. (pip설치먼저 해준다.)
4. sudo python3 [웹 파이썬 파일명] 으로 정적 웹을 실행시켜준다.
   ( \* 이때 중요한 점은 웹 파이썬 파일에서
   app.run(host='0.0.0.0', port=80) 으로 설정해줘야한다. 이유는 80포트는 포트번호가 자동으로 생략되기 때문이다. 만약 정적웹에서 주소에 포트번호를 5000또는 8080으로 일일히 붙여줬다면 해당 포트번호를 입력해준다.)
   <br>
   ![image](https://user-images.githubusercontent.com/98939242/178996187-7d4a97e1-de90-4976-a871-ac2559bebaec.png)
