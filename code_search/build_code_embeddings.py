from code_search import train_model
from code_search import shared
from code_search import utils


def build_code_embeddings():
    for language in shared.LANGUAGES:
        print(f'Building {language} code embeddings')
        model = utils.load_cached_model_weights(language, train_model.get_model())
        code_embedding_predictor = train_model.get_code_embedding_predictor(model)

        evaluation_code_seqs = utils.load_cached_seqs(language, 'evaluation', 'code')
        code_embedding = code_embedding_predictor.predict(evaluation_code_seqs)

        utils.cache_code_embeddings(code_embedding, language)


def build_anns():
    for language in shared.LANGUAGES:
        print(f'Building {language} ann')
        ann = utils.get_annoy_index()

        code_embeddings = utils.load_cached_code_embeddings(language)

        for i in range(code_embeddings.shape[0]):
            ann.add_item(i, code_embeddings[i, :])
        ann.build(60)

        utils.cache_ann(ann, language)


if __name__ == '__main__':
    build_code_embeddings()
    build_anns()
