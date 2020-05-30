import functools
import hashlib
import json
import operator
import re

from django.http import HttpResponseBadRequest, HttpResponse
from django.shortcuts import render, get_object_or_404
from django.core.cache import cache

from code_search import shared, prepare_data, train_model, utils
from code_search_app import models
from code_search_app.shared import get_pygments_html_formatter

RESULTS_PER_LANGUAGE = 10
FILTER_BY_LANGUAGE_REGEX = re.compile(r'\+language=(\S+)')


def get_syntax_highlight_css():
    return get_pygments_html_formatter().get_style_defs('.highlight')


def get_nearest_query_neighbors_per_language(query, languages):
    nearest_neighbors_per_language = {}
    for language in languages:
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


def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def get_user_hash(request):
    ip = get_client_ip(request)
    user_agent = request.META.get('HTTP_USER_AGENT')
    return hashlib.sha1(f'{ip}:{user_agent}'.encode('utf-8')).hexdigest()


def get_filterless_query(query):
    return FILTER_BY_LANGUAGE_REGEX.sub('', query).strip()


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
        'code_documents_count': code_documents_count
    })


def search_view(request):
    if request.method != 'GET':
        return HttpResponseBadRequest('Invalid HTTP method.')

    query = request.GET.get('query')
    if not query or len(query.strip()) == 0:
        return HttpResponseBadRequest('Invalid or missing query.')

    if len(query) > 256:
        return HttpResponseBadRequest('Query too long.')

    models.QueryLog.objects.create(query=query)

    language_filter_match = FILTER_BY_LANGUAGE_REGEX.search(query)
    if language_filter_match is not None:
        languages_match = language_filter_match.group(1).split(',')
        languages = [language.lower() for language in languages_match if language.lower() in shared.LANGUAGES]
        if len(languages) == 0:
            return HttpResponseBadRequest('No valid languages present in the +language filter.')
    else:
        languages = shared.LANGUAGES

    filterless_query = get_filterless_query(query)
    query_hash = hashlib.sha1(query.encode('utf-8')).hexdigest()
    cache_key = f'query:{query_hash}'
    if cache_key in cache:
        nearest_neighbors_per_language = cache.get(cache_key)
    else:
        # Remove the filters from the query
        nearest_neighbors_per_language = get_nearest_query_neighbors_per_language(filterless_query, languages)
        cache.set(cache_key, nearest_neighbors_per_language, timeout=None)  # Never expire

    code_documents_with_distances = []
    for language in languages:
        indices, distances = nearest_neighbors_per_language[language]
        distances_sorted_by_index = map(operator.itemgetter(1), sorted(zip(indices, distances)))
        code_documents = models.CodeDocument.objects.filter(
            language=language, embedded_row_index__in=indices).order_by('embedded_row_index')
        code_documents_with_distances.extend(list(zip(distances_sorted_by_index, code_documents)))

    # Use a custom comparator function to avoid comparing CodeDocument instances in case
    # two distances are equal
    code_documents = sorted(code_documents_with_distances, key=functools.cmp_to_key(lambda a, b: a[0] - b[0]))

    user_hash = get_user_hash(request)
    query_ratings = models.CodeDocumentQueryRating.objects.filter(query=filterless_query, rater_hash=user_hash)
    code_hash_to_rating = {}
    for query_rating in query_ratings:
        code_hash_to_rating[query_rating.code_document.code_hash] = query_rating.rating

    return render(request, 'code_search_app/search.html', {
        'query': query,
        'languages': languages,
        'code_documents': code_documents,
        'syntax_highlight_css': get_syntax_highlight_css(),
        'code_hash_to_rating': code_hash_to_rating
    })


def code_document_visit_view(request, code_hash):
    if request.method != 'POST':
        return HttpResponseBadRequest('Invalid HTTP method.')

    code_document = get_object_or_404(models.CodeDocument, code_hash=code_hash)
    models.CodeDocumentVisitLog.objects.create(code_document=code_document)

    return HttpResponse(status=204)


def code_document_query_rating_view(request, code_hash):
    if request.method != 'POST':
        return HttpResponseBadRequest('Invalid HTTP method.')

    code_document = get_object_or_404(models.CodeDocument, code_hash=code_hash)
    body = json.loads(request.body)

    if 'rating' not in body or 'query' not in body:
        return HttpResponseBadRequest('Invalid POST body.')

    if body['rating'] < 0 or body['rating'] > 3:
        return HttpResponseBadRequest('Rating has to be between 0 and 3.')

    user_hash = get_user_hash(request)
    query_rating, created = models.CodeDocumentQueryRating.objects.get_or_create(
        code_document=code_document, query=body['query'], rater_hash=user_hash, defaults={'rating': body['rating']})

    if not created:
        query_rating.rating = body['rating']
        query_rating.save()

    return HttpResponse(status=204)


def code_document_view(request, code_hash):
    code_document = get_object_or_404(models.CodeDocument, code_hash=code_hash)

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
