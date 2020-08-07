# CodeSnippetSearch

Neural bag of words code search implementation using PyTorch and data from the [CodeSearchNet](https://github.com/github/CodeSearchNet) project.
The model training code was heavily inspired by the baseline (Tensorflow) implementation in the CodeSearchNet repository. 
Currently, Python, Java, Go, Php, Javascript, and Ruby programming languages are supported.

Helpful papers:
* [CodeSearchNet Challenge: Evaluating the State of Semantic Code Search](https://arxiv.org/pdf/1909.09436.pdf)
* [When Deep Learning Met Code Search](https://arxiv.org/pdf/1905.03813.pdf)

## Model description

We are using BPE encoding to encode both code strings and query strings (docstrings are used as a proxy for queries). 
Code strings are padded and encoded to a length of 30 tokens and query strings are padded and encoded to a length of 200 tokens. 
Embedding size is set to 256. Token embeddings are masked and then an unweighted mean is performed to get 256-length vectors for code strings and query strings.
Finally, cosine similarity is calculated between the code vectors and the query vectors and "cosine loss" is calculated 
(the loss function is documented in code_search/train_model.py#cosine_loss).
Further details can be found on the [WANDB run](https://app.wandb.ai/roknovosel/glorified-code-search/runs/21hzzq1h/overview).

# TODO: Documentation is not up to date. It will be rewritten to reflect added repository support
## Model structure

![Model structure](assets/model.png)

## Project structure
- `code_search`: A Python package with scripts to prepare the data, train the language models and save the embeddings
- `code_search_web`: CodeSnippetSearch website Django project
- `cache`: Store for intermediate objects during training (docs, vocabularies, models, embeddings etc.)

## Data

We are using the data from the CodeSearchNet project. Run the following commands to download the required data:

- `$ mkdir -p resources/data; cd resources/data`
- `$ wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{python,java,go,php,javascript,ruby}.zip`

This will download around 20GB of data. Overview of the data structure is listed [here](https://github.com/github/CodeSearchNet/tree/master/resources).

## Populating `env.json`

Copy the `env.template.json` and rename it to `env.json`. If you are planning to just train the models fill out the 
`CODESEARCHNET_RESOURCES_DIR` and `CODESEARCHNET_DATA_DIR` with the paths to the directories you created above.

## Training the models

If you can, you should be performing these steps inside a virtual environment.
To install the required dependencies run: `$ pip install -r requirements.txt`

Before you can start training the models, you will have to add the root folder of this repository to `PYTHONPATH`. You
can export and modify the env variable directly, or you can add a `.pth` file to `site-packages`.
You can find more information on how to this [here](https://stackoverflow.com/questions/10738919/how-do-i-add-a-path-to-pythonpath-in-virtualenv).

### Preparing the data

Data preparation step is separate from the training step because it is time and memory consuming. We will prepare all the
necessary data needed for training. This includes preprocessing code docs, building vocabularies, and encoding sequences.

The first step is to convert evaluation code documents (`*_dedupe_definitions_v2.pkl` files) from a `pickle` format to `jsonl` format. We will be using the jsonl
format throughout the project, since we can read the file line by line and keep the memory footprint minimal. Reading the
evaluation docs requires **more** than 16GB of memory, because the entire file has to be read in memory (largest is `javascript_dedupe_definitions_v2.pkl` at 6.6GB).
If you do not have this kind of horsepower, I suggest renting a cloud server with >16GB of memory and running this step on there. After you are done,
just download the jsonl files to your local machine. Subsequent preparation and training steps should not take more than 16GB of memory.

To convert ruby evaluation docs to `jsonl` format move inside the `code_search` directory run the following command:
`$ python parse_dedupe_definitions.py ruby`. Run this command for the remaining 5 languages: `python`, `java`, `go`, `php` and `javascript`.

To prepare the data for training run: `$ python prepare_data.py --prepare-all`. It uses the Python multiprocessing
module to take advantage of multiple cores. If you encounter memory errors or slow performance you can tweak the number of
processes by changing the parameter passed to `multiprocessing.Pool`.

### Training and evaluation

You start the training by running: `$ python train_model.py`. This will train separate models for each language, build code embeddings
and evaluate them according to MRR (Mean Reciprocal Rank) and output `model_predictions.csv`. These will be evaluated by Github & WANDB 
using NDCG (Normalized Discounted cumulative gain) metric to rank the submissions.

### Query the trained models

Run `$ python search.py "read file lines"` and it will output 3 best ranked results for each language.

# Running the CodeSnippetSearch website locally

- Requirements: A PostgreSQL database
- Fill out the `WEB` object in `env.json` with DB credentials, `SECRET_KEY` and `ALLOWED_HOSTS`
- Run migrations: `$ python manage.py migrate`
- Create cache table: `$ python manage.py createcachetable`
- Import code documents `$ python manage.py import_code_documents`
- Build code embeddings and approximate neighbor search using Annoy (run from the `code_search` directory): `$ python build_code_embeddings.py && python build_anns.py`
- Running the dev server: `$ python manage.py runserver 0.0.0.0:8000`
