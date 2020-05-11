import hashlib

from django.core.management.base import BaseCommand

from code_search_app import models
from code_search import shared, utils


class Command(BaseCommand):
    def handle(self, *args, **options):
        models.CodeDocument.objects.all().delete()
        batch_size = 500
        for language in shared.LANGUAGES:
            print(f'Importing {language} code documents')
            code_docs = []
            for idx, doc in enumerate(utils.load_cached_docs(language, 'evaluation')):
                code_doc = models.CodeDocument(
                    code=doc['function'],
                    code_hash=hashlib.sha1(doc['function'].encode('utf-8')).hexdigest(),
                    url=doc['url'],
                    language=language,
                    repo=doc['nwo'],
                    file_path=doc['path'],
                    identifier=doc['identifier'],
                    embedded_row_index=idx,
                )
                code_docs.append(code_doc)

            models.CodeDocument.objects.bulk_create(code_docs, batch_size=batch_size)
