import hashlib
import tempfile
from typing import List

import torch

from code_search import shared, torch_utils
from code_search.code_embedding import build_code_embeddings, build_annoy_indices
from code_search.model import get_base_language_model_for_evaluation, get_repository_model, \
    get_repository_model_for_evaluation
from code_search.data_manager import DataManager, get_repository_data_manager, get_base_languages_data_manager
from code_search.prepare_data import RepositoryDataPreparer
from code_search.train import train
from code_search.function_parser.extract import extract

from code_search_app.management.commands._utils import download_repository, get_tmp_repository_dir_path, \
    get_repository_commit_hash
from code_search_app import models


def get_code_document_url(organization: str, name: str, commit_hash: str, path: str, start_line: int, end_line: int):
    return f'https://github.com/{organization}/{name}/blob/{commit_hash}/{path}#L{start_line}-L{end_line}'


def extract_repository_language_corpus(data_manager: DataManager, repository_dir: str, language: str):
    data_manager.save_language_corpus(extract(repository_dir, language), language, shared.DataSet.ALL)


def import_corpus(
        data_manager: DataManager,
        repository: models.CodeRepository,
        language: str,
        commit_hash: str,
        batch_size: int = 500):
    corpus = data_manager.get_language_corpus(language, shared.DataSet.ALL)
    code_docs = []
    organization, name = repository.organization, repository.name
    code_language = models.CodeLanguage.objects.get(name=language)
    for idx, doc in enumerate(corpus):
        code_doc = models.CodeDocument(
            url=get_code_document_url(organization, name, commit_hash, doc['path'], doc['start_line'], doc['end_line']),
            path=doc['path'],
            identifier=doc['identifier'],
            code=doc['code'],
            code_hash=hashlib.sha1(doc['code'].encode('utf-8')).hexdigest(),
            embedded_row_index=idx,
            language=code_language,
            repository=repository,
        )
        code_docs.append(code_doc)

    models.CodeDocument.objects.bulk_create(code_docs, batch_size=batch_size)


def build_repository_model(
        repository_data_manager: DataManager,
        base_data_manager: DataManager,
        languages: List[str]):
    base_model = get_base_language_model_for_evaluation(base_data_manager)
    repository_model = get_repository_model(repository_data_manager, languages)

    repository_model.set_query_embedding_weights(
        torch_utils.np_to_torch(repository_data_manager.get_query_embedding_weights()))
    repository_model.set_query_weights_layer(base_model.get_query_weights_layer())

    for language in languages:
        repository_model.set_language_embedding_weights(
            language,
            torch_utils.np_to_torch(repository_data_manager.get_language_embedding_weights(language)))
        repository_model.set_language_weights_layer(language, base_model.get_language_weights_layer(language))

    repository_data_manager.save_torch_model(repository_model)


def train_repository_model(data_manager: DataManager, languages: List[str], device: torch.device):
    model = get_repository_model(data_manager, languages, device)
    train(
        model,
        data_manager,
        languages,
        device,
        learning_rate=shared.LEARNING_RATE,
        batch_size=shared.TRAIN_BATCH_SIZE,
        mrr_eval_batch_size=100,
        verbose=True)


def update_repository(repository: models.CodeRepository):
    repository.update_status = models.CodeRepository.UPDATE_IN_PROGRESS
    repository.save()
    models.CodeDocument.objects.filter(repository=repository).delete()

    organization, name = repository.organization, repository.name
    languages = [language.name for language in repository.languages.all()]

    repository_dir = get_tmp_repository_dir_path(tempfile.TemporaryDirectory(), organization, name)
    download_repository(organization, name, repository_dir)
    repository_commit_hash = get_repository_commit_hash(repository_dir)

    print(f'Repository {organization}/{name} downloaded...')

    repository_data_manager = get_repository_data_manager(organization, name)
    base_data_manager = get_base_languages_data_manager()
    for language in languages:
        print(f'Extracting corpus for {organization}/{name} {language}...')
        extract_repository_language_corpus(repository_data_manager, repository_dir, language)

    # TODO: Check if train corpus size is enough
    data_preparer = RepositoryDataPreparer(repository_data_manager, base_data_manager, languages, verbose=True)
    data_preparer.prepare(
        code_vocabulary_size=shared.CODE_VOCABULARY_SIZE,
        code_pct_bpe=shared.VOCABULARY_PCT_BPE,
        query_vocabulary_size=shared.QUERY_VOCABULARY_SIZE,
        query_pct_bpe=shared.VOCABULARY_PCT_BPE,
        code_seq_max_length=shared.CODE_MAX_SEQ_LENGTH,
        query_seq_max_length=shared.QUERY_MAX_SEQ_LENGTH)

    device = torch_utils.get_device()
    build_repository_model(repository_data_manager, base_data_manager, languages)
    train_repository_model(repository_data_manager, languages, device)

    model = get_repository_model_for_evaluation(repository_data_manager, languages, device)
    build_code_embeddings(model, repository_data_manager, languages, device)
    build_annoy_indices(repository_data_manager, languages, n_trees=600)

    for language in languages:
        import_corpus(repository_data_manager, repository, language, repository_commit_hash)

    repository.commit_hash = repository_commit_hash
    repository.update_status = models.CodeRepository.UPDATE_FINISHED
    repository.save()


def update_repositories(repositories: List[models.CodeRepository]):
    for repository in repositories:
        print(f'Preparing {repository}')
        update_repository(repository)
