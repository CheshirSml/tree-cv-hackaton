import json
from django.db.models import Q
from django.db.models import F
from django.db.models import OuterRef, Subquery
from django.shortcuts import render
from django.views.generic import TemplateView
from django.utils import timezone
from core import serializers as core_serializers

from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.generics import CreateAPIView
from rest_framework.generics import ListAPIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet, ViewSet
from rest_framework.permissions import IsAuthenticated
from rest_framework.permissions import AllowAny
from core.paginators import StandardResultsSetPagination

from rest_framework.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_200_OK,
    HTTP_201_CREATED
)

from .models import Checkup, CheckupPhoto, CheckupPhotoAnnotation
from .serializers import (
    CheckupSerializer,
    CheckupPhotoSerializer,
    CheckupPrototypeSerializer,
)

from core.choices import ObjectType, Breed, Condition, Season, Artifact, CheckupStatus, RUS_TO_ENUM
from .utils.image_utils import get_base64_image
from core.service.llmtunnel_provider import LLMProvider
from django.conf import settings
# from ultralytics import YOLO

# cv_model = YOLO('mlmodels/best_maksim-tree.pt') 

from core.utils.annotator import annotate_photo

class CheckupViewSet(ModelViewSet):
    queryset = Checkup.objects.all().order_by("-report_date")
    serializer_class = CheckupSerializer
    pagination_class = StandardResultsSetPagination

    @action(detail=False, methods=["get"])
    def prototype(self, request):
        """
        Вернёт пустой Checkup (пример структуры для фронта)
        """
        serializer = CheckupPrototypeSerializer(Checkup(
            report_date=timezone.now().date(),
            plot='a12_sokolniki'
        ))
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def finish(self, request, pk=None):
        """
        Завершает обследование.
        """
        checkup = Checkup.objects.get(pk=pk)
        for photo in checkup.photos.all():
            # print(photo)
            photo_base64 = get_base64_image(photo.photo)
            image_size = (1024, 1024)

            provider = LLMProvider(
                settings.PROVIDER_SECRET_KEY,
                settings.PROVIDER_URL,
                settings.PROVIDER_SUBMODEL,
            )
            if not hasattr(photo, 'annotation'):
                photo.annotation = CheckupPhotoAnnotation.objects.create(photo=photo)
            tree = provider.predict(photo_base64, image_size)
            print(tree)
            
            photo.annotation.bbox = tree['bbox']
            photo.annotation.object_type = RUS_TO_ENUM[tree["type"]]
            photo.annotation.breed = RUS_TO_ENUM.get(tree["breed"], Breed.UNKNOWN)
            photo.annotation.condition = RUS_TO_ENUM[tree["condition"]]
            photo.annotation.is_dry = tree["is_dry"]
            photo.annotation.percentage_dried = tree["percentage_dried"]
            photo.annotation.artifacts = [RUS_TO_ENUM[a] for a in tree.get("artifacts", [])]
            photo.annotation.description = tree.get("description", "")
            photo.annotation.season = RUS_TO_ENUM.get(tree["season"])
            photo.annotation.save()

            annotate_photo(photo)


            # Отправить в ЛЛМ, получить ответ
            # Отправить в йолу, сохранить картинку
        
        checkup.status = CheckupStatus.Completed
        checkup.save()
        serializer = self.get_serializer(checkup, many=False)
        return Response(serializer.data)


class CheckupPhotoViewSet(ModelViewSet):
    serializer_class = CheckupPhotoSerializer
    http_method_names = ["post", "delete", "get", "head", "options"]

    def get_queryset(self):
        return CheckupPhoto.objects.all()

    @action(detail=False, methods=["get"], url_path="by-checkup/(?P<checkup_id>[^/.]+)")
    def by_checkup(self, request, checkup_id=None):
        """
        Получить все фото по обследованию
        """
        photos = self.get_queryset().filter(checkup_id=checkup_id)
        serializer = self.get_serializer(photos, many=True)
        return Response(serializer.data)