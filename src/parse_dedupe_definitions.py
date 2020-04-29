import sys
import pickle
import json

file = sys.argv[1]
save = sys.argv[2]

with open(file, 'rb') as f:
    definitions = pickle.load(f)

with open(save, 'w', encoding='utf-8') as f:
    for definition in definitions:
        f.write(json.dumps(
            {
                'url': definition['url'],
                'identifier': definition['identifier'],
                'function_tokens': definition['function_tokens']
            }
        ) + '\n')
