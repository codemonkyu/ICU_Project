from django.db import models

# Create your models here.


class Doorlog(models.Model):
    doorstate = models.CharField(max_length=10, blank=True, null=True)
    opentime = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'doorlog'