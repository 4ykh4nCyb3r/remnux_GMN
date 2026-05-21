import json
import argparse
import os
from sklearn.feature_extraction.text import CountVectorizer

def clean_instruction(inst_string):
    if ":\t" in inst_string:
        return inst_string.split(":\t")[1]
    return inst_string

def get_corpus_and_refs(functions):
    corpus = []
    block_refs = []
    for func in functions:
        func_name = func['function_name']
        for block in func['blocks']:
            cleaned_insts = [clean_instruction(inst) for inst in block['instructions']]
            corpus.append(" ".join(cleaned_insts))
            block_refs.append({'func': func_name, 'block_addr': block['address']})
    return corpus, block_refs

def process_features(input_1, input_2, output_1, output_2):
    print(f"[*] Loading raw graphs from {input_1} and {input_2}...")
    with open(input_1, 'r') as f: funcs_v1 = json.load(f)
    with open(input_2, 'r') as f: funcs_v2 = json.load(f)

    # 1. Extract text from BOTH binaries
    corpus_v1, refs_v1 = get_corpus_and_refs(funcs_v1)
    corpus_v2, refs_v2 = get_corpus_and_refs(funcs_v2)

    # 2. Build the GLOBAL Vocabulary using both corpuses
    print("[*] Building Global AI Vocabulary...")
    global_corpus = corpus_v1 + corpus_v2
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(global_corpus) # The AI learns the master dictionary here
    
    vocab = vectorizer.get_feature_names_out()
    print(f"[*] Global Vocabulary Size: {len(vocab)} unique tokens.")

    # 3. Apply the Global Dictionary to v1
    features_v1 = vectorizer.transform(corpus_v1).toarray()
    for i, ref in enumerate(refs_v1):
        target_func = next(f for f in funcs_v1 if f['function_name'] == ref['func'])
        target_block = next(b for b in target_func['blocks'] if b['address'] == ref['block_addr'])
        target_block['features'] = features_v1[i].tolist()

    # 4. Apply the Global Dictionary to v2
    features_v2 = vectorizer.transform(corpus_v2).toarray()
    for i, ref in enumerate(refs_v2):
        target_func = next(f for f in funcs_v2 if f['function_name'] == ref['func'])
        target_block = next(b for b in target_func['blocks'] if b['address'] == ref['block_addr'])
        target_block['features'] = features_v2[i].tolist()

    # 5. Save the perfectly aligned JSONs
    with open(output_1, 'w') as f: json.dump(funcs_v1, f, indent=4)
    with open(output_2, 'w') as f: json.dump(funcs_v2, f, indent=4)

    print(f"[+] Aligned feature vectors saved to {output_1} and {output_2}!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert assembly to ALIGNED ML feature vectors.")
    parser.add_argument("input_1", help="Raw JSON for v1")
    parser.add_argument("input_2", help="Raw JSON for v2")
    parser.add_argument("-o1", "--output_1", help="Output JSON for v1", required=True)
    parser.add_argument("-o2", "--output_2", help="Output JSON for v2", required=True)
    
    args = parser.parse_args()
    process_features(args.input_1, args.input_2, args.output_1, args.output_2)