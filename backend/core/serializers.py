from rest_framework import serializers
from easy_thumbnails.files import get_thumbnailer

from .models import Checkup, CheckupPhoto, CheckupPhotoAnnotation, DistrictArea
from .choices import (
    RUS_TO_ENUM,
)
# Реверс-маппинг: из EN -> RU
ENUM_TO_RUS = {v: k for k, v in RUS_TO_ENUM.items()}


class CheckupPhotoAnnotationSerializer(serializers.ModelSerializer):

    annotated_photo = serializers.SerializerMethodField()

    @staticmethod
    def get_annotated_photo(obj):
        crop_options = {'size': (250, 250), 'crop': 'scale'}
        try:
            return get_thumbnailer(obj.annotated_photo).get_thumbnail(crop_options).url
        except Exception as e:
            pass

    class Meta:
        model = CheckupPhotoAnnotation
        fields = "__all__"

    def to_representation(self, instance):
        data = super().to_representation(instance)

        # заменяем на русский
        for field in ["object_type", "breed", "condition", "season"]:
            if data.get(field):
                data[field] = ENUM_TO_RUS.get(data[field], data[field])

        if data.get("artifacts"):
            data["artifacts"] = [ENUM_TO_RUS.get(a, a) for a in data["artifacts"]]

        return data

    def to_internal_value(self, data):
        """
        Поддержка входа с русскими значениями
        """
        new_data = data.copy()

        for field in ["object_type", "breed", "condition", "season"]:
            if field in new_data and new_data[field] in RUS_TO_ENUM:
                new_data[field] = RUS_TO_ENUM[new_data[field]]

        if "artifacts" in new_data:
            new_data["artifacts"] = [
                RUS_TO_ENUM.get(a, a) for a in new_data["artifacts"]
            ]

        return super().to_internal_value(new_data)

class CheckupPhotoSerializer(serializers.ModelSerializer):
    annotation = CheckupPhotoAnnotationSerializer(many=False, read_only=True)
    preview = serializers.SerializerMethodField()

    class Meta:
        model = CheckupPhoto
        fields = "__all__"

    def validate_photo(self, value):
        max_size = 10 * 1024 * 1024  # 10 MB
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif']

        if value.size > max_size:
            raise serializers.ValidationError("Размер файла превышает 10 MB.")

        if hasattr(value, "content_type") and value.content_type not in allowed_types:
            raise serializers.ValidationError("Недопустимый тип файла. Разрешены JPG, JPEG, PNG, GIF.")

        return value

    @staticmethod
    def get_preview(obj):
        crop_options = {'size': (450, 450), 'crop': 'scale'}
        try:
            return get_thumbnailer(obj.photo).get_thumbnail(crop_options).url
        except Exception as e:
            pass


class DistrictAreaSerializer(serializers.ModelSerializer):
    class Meta:
        model = DistrictArea
        fields = "__all__"


class CheckupSerializer(serializers.ModelSerializer):

    photos = CheckupPhotoSerializer(many=True, read_only=True)
    area_detail = DistrictAreaSerializer(source='area', read_only=True)

    class Meta:
        model = Checkup
        fields = "__all__"


class CheckupPrototypeSerializer(serializers.ModelSerializer):

    class Meta:
        model = Checkup
        fields = "__all__"
