from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from faceAuthentication.utils import make_classification_model
from .models import User

@api_view(['GET'])  # Changed to list format
def make_the_model(request):

    users = User.objects.all().values('id', 'first_name', 'last_name', 'username', 'created_at')
    users_list = list(users)
        
    model = make_classification_model(len(users_list))
    return Response({"message":"Created the model"})

