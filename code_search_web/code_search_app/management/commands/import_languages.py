from django.core.management.base import BaseCommand

from code_search_app import models
from code_search import shared


class Command(BaseCommand):
    def handle(self, *args, **options):
        models.CodeLanguage.objects.all().delete()

        for language in shared.LANGUAGES:
            models.CodeLanguage.objects.create(name=language)
