from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def cctvvideo(request):
    return render(request,'cctv_video.html')

def cctvrecord(request):
    return render(request, 'cctv_record.html')

