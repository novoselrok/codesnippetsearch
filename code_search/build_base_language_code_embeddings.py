import torch
import numpy as np

from code_search import train, utils, shared
from code_search.train import CodeSearchNet


def batch_predict(model, language, batch_size=1000):
    language_dir = utils.get_base_language_serialized_data_path(language)
    code_seqs = utils.load_serialized_seqs(language_dir, 'code')
    n_seqs = code_seqs.shape[0]
    code_embeddings = np.zeros((n_seqs, shared.EMBEDDING_SIZE))

    idx = 0
    for _ in range((n_seqs // batch_size) + 1):
        end_idx = min(n_seqs, idx + batch_size)
        batch_code_seqs = torch.from_numpy(code_seqs[idx:end_idx, :]).to(train.device)
        code_embeddings[idx:end_idx, :] = model.encode_code(language, batch_code_seqs).detach().cpu().numpy()
        idx += batch_size

    return code_embeddings


def build_code_embeddings():
    model: CodeSearchNet = train.get_model().to(train.device)
    model.load_state_dict(utils.load_serialized_model(shared.BASE_LANGUAGES_DIR))
    model.eval()
    for language in shared.LANGUAGES:
        language_dir = utils.get_base_language_serialized_data_path(language)
        print(f'Building {language} code embeddings')
        utils.serialize_code_embeddings(batch_predict(model, language), language_dir)


def build_ann():
    for language in shared.LANGUAGES:
        print(language)
        language_dir = utils.get_base_language_serialized_data_path(language)
        code_embeddings = utils.load_serialized_code_embeddings(language_dir)

        ann = utils.get_annoy_index()
        for i in range(code_embeddings.shape[0]):
            ann.add_item(i, code_embeddings[i, :])
        ann.build(1000)

        utils.serialize_ann(ann, language_dir)


if __name__ == '__main__':
    build_code_embeddings()
    build_ann()
