import hashlib
import json
import operator
import re
import os
from typing import List, Tuple, Any, Dict

from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt
from pygments import highlight
from pygments.lexers import get_lexer_by_name

from code_search_app import models
from code_search_app.shared import get_pygments_html_formatter

from code_search import shared
from code_search.torch_utils import get_device
from code_search.search import get_nearest_embedding_neighbors_per_language, get_nearest_code_neighbors,\
    get_code_embedding, get_query_embedding
from code_search.data_manager import get_repository_data_manager
from code_search.model import get_repository_model_for_evaluation


RESULTS_PER_LANGUAGE = 20
FILTER_BY_LANGUAGE_REGEX = re.compile(r'\+language=(\S+)')


def syntax_highlight(code_document: models.CodeDocument):
    lexer = get_lexer_by_name(code_document.language.name, startinline=True)
    formatter = get_pygments_html_formatter()
    return highlight(code_document.code, lexer, formatter)


def code_repository_as_json(code_repository: models.CodeRepository) -> Dict[str, Any]:
    return {
        'id': code_repository.id,
        'organization': code_repository.organization,
        'name': code_repository.name,
        'commitHash': code_repository.commit_hash,
        'description': code_repository.description,
        'languages': [language.name for language in code_repository.languages.all()]
    }


def code_document_as_json(code_document: models.CodeDocument) -> Dict[str, Any]:
    return {
        'filename': os.path.basename(code_document.path),
        'url': code_document.url,
        'codeHtml': syntax_highlight(code_document),
        'codeHash': code_document.code_hash,
        'language': code_document.language.name
    }


def code_documents_with_distances_as_json(
        code_documents_with_distances: List[Tuple[float, models.CodeDocument]]) -> List[Dict[str, Any]]:
    code_documents = []
    for distance, code_document in code_documents_with_distances:
        code_documents.append({
            'distance': distance,
            **code_document_as_json(code_document)
        })
    return code_documents


def get_filterless_query(query):
    return FILTER_BY_LANGUAGE_REGEX.sub('', query).strip()


def get_code_documents_from_indices(
        repository: models.CodeRepository, language: str, indices: List[int]) -> List[models.CodeDocument]:
    return models.CodeDocument.objects.filter(
        repository=repository,
        language__name=language,
        embedded_row_index__in=indices).order_by('embedded_row_index')


def sort_code_documents_with_distances_by_index(
        code_documents: List[models.CodeDocument],
        indices: List[int],
        distances: List[float]) -> List[Tuple[float, int, models.CodeDocument]]:
    distances_sorted_by_index = map(operator.itemgetter(1), sorted(zip(indices, distances)))
    return list(zip(distances_sorted_by_index, sorted(indices), code_documents))


def sort_code_documents_by_distance(
        code_documents_with_distances: List[
            Tuple[float, int, models.CodeDocument]]) -> List[Tuple[float, models.CodeDocument]]:
    # Sort by distance, use indices as tie breakers
    return list(map(lambda _: (_[0], _[2]), sorted(code_documents_with_distances, key=lambda _: _[:2])))


def api_repositories_view(request):
    if request.method != 'GET':
        return HttpResponseBadRequest('Invalid HTTP method.')

    code_repositories = models.CodeRepository.objects.filter(update_status=models.CodeRepository.UPDATE_FINISHED)
    return JsonResponse({
        'codeRepositories': [code_repository_as_json(code_repository) for code_repository in code_repositories]})


def api_repository_view(request, repository_organization, repository_name):
    repository = get_object_or_404(models.CodeRepository, organization=repository_organization, name=repository_name)
    if repository.update_status != models.CodeRepository.UPDATE_FINISHED:
        return HttpResponseBadRequest('Repository update is not finished.')

    return JsonResponse(code_repository_as_json(repository))


def api_repository_search_view(request, repository_organization, repository_name):
    if request.method != 'GET':
        return HttpResponseBadRequest('Invalid HTTP method.')

    query = request.GET.get('query')
    if not query or len(query.strip()) == 0:
        return HttpResponseBadRequest('Invalid or missing query.')

    if len(query) > 256:
        return HttpResponseBadRequest('Query too long.')

    models.QueryLog.objects.create(query=query)

    repository = get_object_or_404(models.CodeRepository, organization=repository_organization, name=repository_name)
    repository_languages = [language.name for language in repository.languages.all()]

    language_filter_match = FILTER_BY_LANGUAGE_REGEX.search(query)
    if language_filter_match is not None:
        languages_match = language_filter_match.group(1).split(',')
        languages = [language.lower() for language in languages_match if language.lower() in repository_languages]
        if len(languages) == 0:
            return HttpResponseBadRequest('No valid languages present in the +language filter.')
    else:
        languages = repository_languages

    cache_key = hashlib.sha1(f'query:{repository_organization}:{repository_name}:{query}'.encode('utf-8')).hexdigest()
    if cache_key in cache:
        nearest_neighbors_per_language = cache.get(cache_key)
    else:
        data_manager = get_repository_data_manager(repository.organization, repository.name)
        device = get_device()
        model = get_repository_model_for_evaluation(data_manager, languages, device)
        query_embedding = get_query_embedding(
            model, data_manager, get_filterless_query(query), shared.QUERY_MAX_SEQ_LENGTH, device)
        nearest_neighbors_per_language = get_nearest_embedding_neighbors_per_language(
            data_manager,
            languages,
            query_embedding,
            results_per_language=RESULTS_PER_LANGUAGE)
        cache.set(cache_key, nearest_neighbors_per_language, timeout=None)  # Never expire

    code_documents_with_distances = []
    for language in languages:
        indices, distances = nearest_neighbors_per_language[language]
        code_documents = get_code_documents_from_indices(repository, language, indices)
        code_documents_with_distances.extend(
            sort_code_documents_with_distances_by_index(code_documents, indices, distances))

    code_documents_with_distances = sort_code_documents_by_distance(code_documents_with_distances)
    return JsonResponse({'codeDocuments': code_documents_with_distances_as_json(code_documents_with_distances)})


@csrf_exempt
def api_repository_search_by_code_view(request, repository_organization, repository_name):
    if request.method != 'POST':
        return HttpResponseBadRequest('Invalid HTTP method.')

    body = json.loads(request.body)
    code = body.get('code')
    if not code or len(code.strip()) == 0:
        return HttpResponseBadRequest('Invalid or missing code.')

    if len(code) > 4096:
        return HttpResponseBadRequest('Code too long.')

    language = body.get('language')
    repository = get_object_or_404(models.CodeRepository, organization=repository_organization, name=repository_name)
    repository_languages = [language.name for language in repository.languages.all()]

    if language not in repository_languages:
        return HttpResponseBadRequest(f'{language} is not a valid repository language.')

    data_manager = get_repository_data_manager(repository.organization, repository.name)
    device = get_device()
    model = get_repository_model_for_evaluation(data_manager, repository_languages, device)
    code_embedding = get_code_embedding(
        model, data_manager, code, language, shared.CODE_MAX_SEQ_LENGTH, device)

    indices, distances = get_nearest_embedding_neighbors_per_language(
        data_manager,
        [language],
        code_embedding,
        results_per_language=RESULTS_PER_LANGUAGE)[language]

    code_documents = get_code_documents_from_indices(repository, language, indices)
    code_documents_with_distances = sort_code_documents_by_distance(
        sort_code_documents_with_distances_by_index(code_documents, indices, distances))

    return JsonResponse({'codeDocuments': code_documents_with_distances_as_json(code_documents_with_distances)})


def api_code_document_view(request, repository_organization, repository_name, code_hash):
    if request.method != 'GET':
        return HttpResponseBadRequest('Invalid HTTP method.')

    code_document = get_object_or_404(
        models.CodeDocument,
        repository__organization=repository_organization,
        repository__name=repository_name,
        code_hash=code_hash)

    return JsonResponse({'codeDocument': code_document_as_json(code_document)})


def api_similar_code_documents_view(request, repository_organization, repository_name, code_hash):
    if request.method != 'GET':
        return HttpResponseBadRequest('Invalid HTTP method.')

    code_document = get_object_or_404(
        models.CodeDocument,
        repository__organization=repository_organization,
        repository__name=repository_name,
        code_hash=code_hash)
    language = code_document.language.name

    cache_key = hashlib.sha1(
        f'similar_code_documents:{repository_organization}:{repository_name}:{code_hash}'.encode('utf-8')).hexdigest()
    if cache_key in cache:
        indices, distances = cache.get(cache_key)
    else:
        data_manager = get_repository_data_manager(code_document.repository.organization, code_document.repository.name)
        indices, distances = get_nearest_code_neighbors(
            data_manager,
            language,
            code_document.embedded_row_index,
            shared.EMBEDDING_SIZE,
            n_results=RESULTS_PER_LANGUAGE)
        cache.set(cache_key, (indices, distances), timeout=None)

    code_documents = get_code_documents_from_indices(code_document.repository, language, indices)
    code_documents_with_distances = sort_code_documents_by_distance(
        sort_code_documents_with_distances_by_index(code_documents, indices, distances))

    return JsonResponse({'codeDocuments': code_documents_with_distances_as_json(code_documents_with_distances)})
