from code_search import shared, utils


def build_anns():
    for language in shared.LANGUAGES:
        print(f'Building {language} ann')
        ann = utils.get_annoy_index()

        code_embeddings = utils.load_cached_code_embeddings(language)

        for i in range(code_embeddings.shape[0]):
            ann.add_item(i, code_embeddings[i, :])
        ann.build(10)

        utils.cache_ann(ann, language)


if __name__ == '__main__':
    build_anns()
