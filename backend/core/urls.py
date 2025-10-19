from django.urls import path, include
from core import views
from rest_framework.routers import DefaultRouter
from .views import CheckupViewSet, CheckupPhotoViewSet, DistrictAreaViewSet

app_name = 'core'

router = DefaultRouter()
router.register("checkups", CheckupViewSet, basename="checkup")
router.register("photos", CheckupPhotoViewSet, basename="photo")
router.register("areas", DistrictAreaViewSet, basename="area")

urlpatterns = [
    path("", include(router.urls)),
]
