from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name = 'home'),
    path('result',views.prediction, name = 'result')
    # path('history/',views.show_history, name = 'history')
    
]
