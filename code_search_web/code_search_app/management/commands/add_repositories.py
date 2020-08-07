import csv

from django.core.management.base import BaseCommand

from code_search_app import models


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('repositories_csv_file', type=str)

    def handle(self, *args, **options):
        with open(options['repositories_csv_file'], encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                repository = models.CodeRepository.objects.create(
                    organization=row['organization'], name=row['name'], description=row['description'])
                languages = row['languages'].split('|')

                for language in languages:
                    repository.languages.add(models.CodeLanguage.objects.get(name=language))

                self.stdout.write(self.style.SUCCESS(f'Created {repository}'))
