from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def cctvvideo(request):
    return render(request,'cctv_video.html')

def cctvvideorecord(request):
    return render(request,'cctv_videorecord.html')

