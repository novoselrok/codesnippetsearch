from django.core.management.base import BaseCommand

from code_search_app import models
from code_search_app.management.commands._update_repositories import update_repositories


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('repositories', nargs='*', type=str)

    def handle(self, *args, **options):
        if len(options['repositories']) == 0:
            repositories = models.CodeRepository.objects.all()
        else:
            parsed_repositories = [repository.split('/') for repository in options['repositories']]
            repositories = [models.CodeRepository.objects.get(organization=organization, name=name)
                            for organization, name in parsed_repositories]

        update_repositories(repositories)
