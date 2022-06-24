from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

#실시간 영상확인
def cctvweb(request):
    return render(request,'cctv_web.html')

#저장용 실시간 영상
def cctvweb2(request):
    return render(request, 'cctv_web2.html')

#저장된 영상확인
def cctvrecord(request):
    return render(request, 'cctv_record.html')



