"""
URL configuration for faceAuthentication project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from faceAuthentication.api_methods import train
from faceAuthentication.api_methods import register,authenticate

urlpatterns = [
    path('admin/', admin.site.urls),
    path('makeModel/',train.make_the_model,name="Test the API"),
    path('register', register.register_user, name='register_user'),
    path('authenticate', authenticate.authenticate_user, name='authenticate_user')
]
