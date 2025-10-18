from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from core.choices import ObjectType, Breed, Condition, Season, Artifact, CheckupStatus
from django.contrib.postgres.fields import ArrayField


class DistrictArea(models.Model):
    title = models.CharField(max_length=255)
    # coords: [x1, y1, x2, y2] - две точки квадрата
    coords = ArrayField(
        base_field=models.FloatField(),
        size=4,
        help_text="Coordinates of two opposite corners: [x1, y1, x2, y2]"
    )

    def __str__(self):
        return self.title


class Checkup(models.Model):

    area = models.ForeignKey(
        'DistrictArea',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='checkups'
    )
    report_date = models.DateField()
    status = models.CharField(max_length=32, choices=CheckupStatus.choices, default=CheckupStatus.Pending)
    comment = models.TextField(blank=True)

    def __str__(self):
        area_title = self.area.__str__() if self.area else ''
        return f"Checkup {self.id} — {area_title} ({self.get_status_display()})"
    

class CheckupPhoto(models.Model):
    checkup = models.ForeignKey(Checkup, on_delete=models.CASCADE, related_name="photos")
    photo = models.ImageField(upload_to="checkups/originals")
    created_at = models.DateTimeField(auto_now_add=True)

    coords = ArrayField(
        base_field=models.FloatField(),
        size=2,
        help_text="Coordinate: [x, y]",
        null=True,
        default=None,
        blank=True,
    )

    def __str__(self):
        return f"Photo {self.id} for Checkup {self.checkup_id}"


class CheckupPhotoAnnotation(models.Model):
    """
    Один распознанный объект (дерево/кустарник) на фото обследования
    """

    photo = models.OneToOneField("CheckupPhoto", on_delete=models.CASCADE, related_name="annotation")
    annotated_photo = models.ImageField(upload_to="checkups/annotated", blank=True, null=True)

    # bbox: [x1, y1, x2, y2]
    bbox = ArrayField(models.FloatField(), size=4, null=True)

    object_type = models.CharField(max_length=16, choices=ObjectType.choices)
    breed = models.CharField(max_length=32, choices=Breed.choices, default=Breed.UNKNOWN)
    condition = models.CharField(max_length=32, choices=Condition.choices, default=Condition.NORMAL)
    is_dry = models.BooleanField(default=False)
    percentage_dried = models.PositiveSmallIntegerField(default=0)

    artifacts = ArrayField(models.CharField(max_length=32, choices=Artifact.choices), blank=True, default=list)
    description = models.TextField(blank=True)
    season = models.CharField(max_length=32, choices=Season.choices, null=True, blank=True)
    is_tree_finded = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.get_breed_display()} ({self.get_condition_display()})"
