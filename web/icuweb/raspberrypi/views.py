from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from django.shortcuts import render, redirect
from django.views import View
from django_request_mapping import request_mapping
from .models import Doorlog
from django.core.paginator import Paginator
from django.utils import timezone


#실시간 영상확인
def cctvweb(request):
    return render(request,'cctv_web.html')

#저장용 실시간 영상
def cctvweb2(request):
    return render(request, 'cctv_web2.html')

#저장된 영상확인
def cctvrecord(request):
    return render(request, 'cctv_record.html')

#문열림 로그 확인
def cctvdoor(request):
    return render(request, 'cctv_door.html')


@request_mapping("")
class MyView(View):

    @request_mapping("/cctvdoor", method="get")
    def home(self,request):
        doordata = Doorlog.objects.all()
        page = request.GET.get('page', '1')
        question_list = Doorlog.objects.order_by('-id')
        paginator = Paginator(question_list,1000000)
        page_obj = paginator.get_page(page)
        context = {'door': doordata,
                   'question_list': page_obj,
                   'page': page}

        return render(request,'cctv_door.html', context);

    @request_mapping("/hong", method="get")
    def hong(self, request):

        i = timezone.now()
    
        Doorlog(doorstate="열림", opentime = i).save()

        return redirect('/cctvdoor')
    
    
