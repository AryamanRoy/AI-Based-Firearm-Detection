# Generated by Django 4.2.6 on 2023-12-28 16:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app1', '0002_imagemodel_address'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodel',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AlterField(
            model_name='imagemodel',
            name='address',
            field=models.CharField(max_length=255),
        ),
    ]
