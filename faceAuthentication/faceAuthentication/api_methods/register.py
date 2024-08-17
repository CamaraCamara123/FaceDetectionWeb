from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import User
from faceAuthentication.utils import register_user

@api_view(['POST'])
def register(request):
    # Appel de la fonction pour enregistrer l'utilisateur
    register_user(request)

