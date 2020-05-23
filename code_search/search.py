import sys

from sklearn.neighbors import NearestNeighbors

from code_search import shared
from code_search import utils
from code_search import prepare_data
from code_search import train_model

query = sys.argv[1]

for language in shared.LANGUAGES:
    print(f'Evaluating {language}')

    evaluation_docs = [{'url': doc['url'], 'identifier': doc['identifier']}
                       for doc in utils.load_cached_docs(language, 'evaluation')]

    code_embeddings = utils.load_cached_code_embeddings(language)

    query_seqs = prepare_data.pad_encode_seqs(
        prepare_data.preprocess_query_tokens,
        lambda: (line.split(' ') for line in [query]),
        shared.QUERY_MAX_SEQ_LENGTH,
        language,
        'query')

    model = utils.load_cached_model_weights(language, train_model.get_model())
    query_embedding_predictor = train_model.get_query_embedding_predictor(model)
    query_embeddings = query_embedding_predictor.predict(query_seqs)

    nn = NearestNeighbors(n_neighbors=3, metric='cosine', n_jobs=-1)
    nn.fit(code_embeddings)
    _, nearest_neighbor_indices = nn.kneighbors(query_embeddings)

    for query_nearest_code_idx in nearest_neighbor_indices[0, :]:
        print(evaluation_docs[query_nearest_code_idx]['url'])
