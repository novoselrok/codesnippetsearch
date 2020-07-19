from django.core.management.base import BaseCommand

from code_search_app import models


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('organization', type=str)
        parser.add_argument('name', type=str)
        parser.add_argument('languages', type=str)

    def handle(self, *args, **options):
        repository = models.CodeRepository.objects.create(organization=options['organization'], name=options['name'])
        languages = options['languages'].split(',')

        for language in languages:
            repository.languages.add(models.CodeLanguage.objects.get(name=language))

        self.stdout.write(self.style.SUCCESS(f'Created {repository}'))
