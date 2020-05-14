import functools
import hashlib
import operator
import random

from django.http import HttpResponseBadRequest
from django.shortcuts import render, get_object_or_404
from django.core.cache import cache

from code_search import shared, prepare_data, train_model, utils
from code_search_app import models
from code_search_app.shared import get_pygments_html_formatter

RESULTS_PER_LANGUAGE = 10
RANDOM_SNIPPETS = 3


def get_syntax_highlight_css():
    return get_pygments_html_formatter().get_style_defs('.highlight')


def get_nearest_query_neighbors_per_language(query):
    nearest_neighbors_per_language = {}
    for language in shared.LANGUAGES:
        query_seq = prepare_data.pad_encode_query(query, language)

        model = utils.load_cached_model_weights(language, train_model.get_model())
        query_embedding_predictor = train_model.get_query_embedding_predictor(model)
        query_embedding = query_embedding_predictor.predict(query_seq.reshape(1, -1))[0, :]

        ann = utils.load_cached_ann(language)
        nearest_neighbors_per_language[language] = ann.get_nns_by_vector(
            query_embedding, RESULTS_PER_LANGUAGE, include_distances=True)
    return nearest_neighbors_per_language


def get_nearest_code_neighbors(language, embedding_row_index):
    code_language_ann = utils.load_cached_ann(language)
    indices, distances = code_language_ann.get_nns_by_item(
        embedding_row_index, RESULTS_PER_LANGUAGE + 1, include_distances=True)
    return indices[1:], distances[1:]  # Exclude the first result since it belongs to the embedding row


def index_view(request):
    if request.method != 'GET':
        return HttpResponseBadRequest('Invalid HTTP method.')

    code_documents_count_cache_key = 'code_documents_cache_key'
    if code_documents_count_cache_key in cache:
        code_documents_count = cache.get(code_documents_count_cache_key)
    else:
        code_documents_count = models.CodeDocument.objects.count()
        cache.set(code_documents_count_cache_key, code_documents_count, timeout=None)  # Never expire

    return render(request, 'code_search_app/index.html', {
        'code_documents_count': code_documents_count,
    })


def search_view(request):
    if request.method != 'GET':
        return HttpResponseBadRequest('Invalid HTTP method.')

    query = request.GET.get('query')
    if not query or len(query.strip()) == 0:
        return HttpResponseBadRequest('Invalid or missing query.')

    models.QueryLog.objects.create(query=query)

    query_hash = hashlib.sha1(query.encode('utf-8')).hexdigest()
    cache_key = f'query:{query_hash}'
    if cache_key in cache:
        nearest_neighbors_per_language = cache.get(cache_key)
    else:
        nearest_neighbors_per_language = get_nearest_query_neighbors_per_language(query)
        cache.set(cache_key, nearest_neighbors_per_language, timeout=None)  # Never expire

    code_documents_with_distances = []
    for language in shared.LANGUAGES:
        indices, distances = nearest_neighbors_per_language[language]
        distances_sorted_by_index = map(operator.itemgetter(1), sorted(zip(indices, distances)))
        code_documents = models.CodeDocument.objects.filter(
            language=language, embedded_row_index__in=indices).order_by('embedded_row_index')
        code_documents_with_distances.extend(list(zip(distances_sorted_by_index, code_documents)))

    # Use a custom comparator function to avoid comparing CodeDocument instances in case
    # two distances are equal
    code_documents = sorted(code_documents_with_distances, key=functools.cmp_to_key(lambda a, b: a[0] - b[0]))
    return render(request, 'code_search_app/search.html', {
        'query': query,
        'languages': shared.LANGUAGES,
        'code_documents': code_documents,
        'syntax_highlight_css': get_syntax_highlight_css()
    })


def code_document_view(request, code_hash):
    code_document = get_object_or_404(models.CodeDocument, code_hash=code_hash)
    models.CodeDocumentVisitLog.objects.create(code_document=code_document)

    cache_key = f'similar_code_documents:{code_hash}'
    if cache_key in cache:
        indices, distances = cache.get(cache_key)
    else:
        indices, distances = get_nearest_code_neighbors(code_document.language, code_document.embedded_row_index)
        cache.set(cache_key, (indices, distances), timeout=None)

    distances_sorted_by_index = map(operator.itemgetter(1), sorted(zip(indices, distances)))
    code_documents = models.CodeDocument.objects.filter(
        language=code_document.language, embedded_row_index__in=indices).order_by('embedded_row_index')
    code_documents_with_distances = zip(distances_sorted_by_index, code_documents)
    code_documents = sorted(code_documents_with_distances, key=functools.cmp_to_key(lambda a, b: a[0] - b[0]))

    return render(request, 'code_search_app/similar_code_documents.html', {
        'code_document': code_document,
        'similar_code_documents': code_documents,
        'syntax_highlight_css': get_syntax_highlight_css()
    })
