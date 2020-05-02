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


if __name__ == '__main__':
    build_code_embeddings()
