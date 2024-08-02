import json

import numpy as np
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_trec_covid():
    searcher = LuceneSearcher("/tmp/data/covid/covid_index")

    # load topics
    with open('/tmp/data/covid/queries.jsonl', 'r') as f:
        topics = {json.loads(line)['_id']: json.loads(line) for line in f}

    # Load qrels
    qrels = {}
    with open('/tmp/data/covid/qrels/test.tsv', 'r') as f:
        next(f)  # Skip header
        for line in f:
            query_id, doc_id, relevance = line.strip().split('\t')
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(relevance)

    return searcher, topics, qrels


def split_dataset(topics, qrels, test_size=0.2, val_size=0.2):
    topic_ids = list(topics.keys())
    train_val_ids, test_ids = train_test_split(topic_ids, test_size=test_size, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size / (1 - test_size), random_state=42)

    return {
        'train': {tid: topics[tid] for tid in train_ids},
        'val': {tid: topics[tid] for tid in val_ids},
        'test': {tid: topics[tid] for tid in test_ids}
    }, {
        'train': {tid: qrels[tid] for tid in train_ids},
        'val': {tid: qrels[tid] for tid in val_ids},
        'test': {tid: qrels[tid] for tid in test_ids}
    }


# Calculate Average Precision
def average_precision_at_k(relevant_docs, retrieved_docs, k=10):
    relevant_retrieved = 0
    sum_precision = 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            relevant_retrieved += 1
            sum_precision += relevant_retrieved / (i + 1)
    return sum_precision / min(len(relevant_docs), k) if relevant_docs else 0


# Evaluate BM25 performance using MAP@10
def evaluate_bm25(searcher, topics, qrels, k1, b):
    searcher.set_bm25(k1, b)
    ap_scores = []

    for topic_id, topic in topics.items():
        query = topic['text']
        hits = searcher.search(query, k=10)  # Only retrieve top 10 documents
        retrieved_docs = [hit.docid for hit in hits]
        relevant_docs = set(qrels[topic_id].keys())
        ap = average_precision_at_k(relevant_docs, retrieved_docs, k=10)
        ap_scores.append(ap)

    return np.mean(ap_scores)  # This is the MAP@10 score


# Grid search for optimal BM25 parameters
def grid_search(searcher, train_topics, train_qrels, val_topics, val_qrels):
    k1_range = np.arange(0.0, 2.0, 0.1)
    b_range = np.arange(0.0, 1.0, 0.1)

    results = []
    grid_data = np.zeros((len(k1_range), len(b_range)))

    for i, k1 in enumerate(tqdm(k1_range, desc="Grid Search Progress")):
        for j, b in enumerate(b_range):
            map_score = evaluate_bm25(searcher, val_topics, val_qrels, k1, b)
            results.append({'k1': k1, 'b': b, 'map@10': map_score})
            grid_data[i, j] = map_score

    results_df = pd.DataFrame(results)
    best_params = results_df.loc[results_df['map@10'].idxmax()]
    return best_params['k1'], best_params['b'], results_df, grid_data, k1_range, b_range


def save_origin_heatmap_data(grid_data, k1_range, b_range, filename='bm25_heatmap_data.txt'):
    with open(filename, 'w') as f:
        # Write header
        f.write("k1\\b\t" + "\t".join(map(str, b_range)) + "\n")

        # Write data
        for i, k1 in enumerate(k1_range):
            row_data = [str(k1)] + [str(val) for val in grid_data[i, :]]
            f.write("\t".join(row_data) + "\n")

    print(f"Heatmap data saved to {filename}")


# Main execution
def main():
    print("Loading TREC-COVID dataset from BEIR...")
    searcher, topics, qrels = load_trec_covid()

    print("Splitting dataset...")
    split_topics, split_qrels = split_dataset(topics, qrels)

    print("Performing grid search...")
    best_k1, best_b, results_df, grid_data, k1_range, b_range = grid_search(
        searcher, split_topics['train'], split_qrels['train'],
        split_topics['val'], split_qrels['val']
    )

    print(f"Best parameters: k1={best_k1}, b={best_b}")

    print("Evaluating on test set...")
    test_map = evaluate_bm25(searcher, split_topics['test'], split_qrels['test'], best_k1, best_b)
    print(f"Test set MAP: {test_map}")

    # Save results
    results_df.to_csv('bm25_grid_search_results_map10.csv', index=False)
    with open('best_params_map10.json', 'w') as f:
        json.dump({'k1': best_k1, 'b': best_b, 'test_map@10': test_map}, f)

    # Save heatmap data for Origin
    save_origin_heatmap_data(grid_data, k1_range, b_range)


if __name__ == '__main__':
    main()
