from django.db import models


class User(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    username = models.CharField(unique=True, max_length=100)
    pass_phrase = models.CharField(max_length=100)

    def __str__(self):
        return f'{self.first_name} {self.last_name}'
