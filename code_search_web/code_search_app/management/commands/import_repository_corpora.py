from django.core.management.base import BaseCommand
from code_search.data_manager import get_repository_data_manager

from code_search_app import models
from code_search_app.management.commands._update_repositories import import_corpora


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

        for repository in repositories:
            repository_data_manager = get_repository_data_manager(repository.organization, repository.name)
            languages = [language.name for language in repository.languages.all()]
            import_corpora(repository_data_manager, repository, languages, repository.commit_hash)
