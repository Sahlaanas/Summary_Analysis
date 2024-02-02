from django.db import models

# Create your models here.

class history(models.Model):
    prompttext=models.TextField()
    summarytext=models.TextField()
    content = models.DecimalField(max_digits=10, decimal_places=3)
    wording = models.DecimalField(max_digits=10, decimal_places=3)
    responses = models.TextField(blank=True)
 