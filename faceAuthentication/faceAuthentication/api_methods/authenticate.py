from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import User
from faceAuthentication.utils import authenticate_user

@api_view(['POST'])
def authenticate(request):
    # Appel de la fonction pour enregistrer l'utilisateur
    authenticate_user(request)

