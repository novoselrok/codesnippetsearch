import tempfile

from django.core.management.base import BaseCommand

from code_search.data_manager import get_repository_data_manager

from code_search_app import models
from code_search_app.management.commands._update_repositories import extract_repository_language_corpus
from code_search_app.management.commands._utils import download_repository, get_tmp_repository_dir_path


class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument('organization', type=str)
        parser.add_argument('name', type=str)
        parser.add_argument('language', type=str)

    def handle(self, *args, **options):
        repository = models.CodeRepository.objects.get(organization=options['organization'], name=options['name'])
        language = options['language']

        repository_data_manager = get_repository_data_manager(repository.organization, repository.name)
        repository_dir = get_tmp_repository_dir_path(
            tempfile.TemporaryDirectory(), repository.organization, repository.name)
        download_repository(repository.organization, repository.name, repository_dir)
        extract_repository_language_corpus(repository_data_manager, repository_dir, language)
