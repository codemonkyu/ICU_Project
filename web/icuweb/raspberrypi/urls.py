from django_request_mapping import UrlPattern
from raspberrypi.views import MyView
from . import views
from django.urls import path, include


# urlpatterns = [
#     path('/cctvdoor',MyView.as_view(), name='cctvdoor'),
# ]

urlpatterns = UrlPattern();
urlpatterns.register(MyView);