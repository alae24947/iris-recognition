from django.db import models
from django.contrib.auth.models import User
import torch.nn as nn
from torchvision import models as tv_models

# Structure de l'IA (Copie de ton notebook)
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = tv_models.resnet18(weights=None)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
from django.db import models
from django.contrib.auth.models import User

# Votre structure SiameseNetwork reste inchangée au-dessus...

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # Nouveau champ pour le numéro de téléphone
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    iris_image = models.ImageField(upload_to='iris_auth/')
    balance = models.FloatField(default=1000.0)

    def __str__(self):
        return f"{self.user.first_name} ({self.user.email})"