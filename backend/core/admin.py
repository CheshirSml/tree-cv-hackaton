from django.contrib import admin
from core.models import Checkup, CheckupPhoto, CheckupPhotoAnnotation, DistrictArea
from django.db.models import Count


@admin.register(Checkup)
class PrescriptionAdmin(admin.ModelAdmin):
    list_display = [ 'area', 'report_date', 'status', 'photos_count' ]

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        queryset = queryset.annotate(_photos_count=Count('photos'))
        return queryset
    
    def photos_count(self, obj):
        return obj._photos_count

    photos_count.short_description = 'Количество фото'

@admin.register(CheckupPhoto)
class CheckupPhotoAdmin(admin.ModelAdmin):
    list_display = [ 'created_at', 'checkup' ]


@admin.register(CheckupPhotoAnnotation)
class CheckupPhotoAnnotationAdmin(admin.ModelAdmin):
    list_display = [ 'id', 'breed' ]


@admin.register(DistrictArea)
class DistrictAreaAdmin(admin.ModelAdmin):
    list_display = ("title", "coords")
