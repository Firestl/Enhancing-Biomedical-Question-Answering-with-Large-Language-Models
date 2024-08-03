# BioASQ BM25 Parameter Optimization
# This script optimizes BM25 parameters for the BioASQ dataset using grid search.

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyserini.search import LuceneSearcher
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
DATA_DIR = "/data/bioasq"
INDEX_DIR = os.path.join(DATA_DIR, "index")
QUERIES_FILE = os.path.join(DATA_DIR, "queries.json")
QRELS_FILE = os.path.join(DATA_DIR, "qrels.txt")

# Type aliases for improved readability
TopicDict = Dict[str, Dict[str, str]]
QrelsDict = Dict[str, Dict[str, int]]


def load_bioasq_dataset() -> Tuple[LuceneSearcher, TopicDict, QrelsDict]:
    """
    Load the BioASQ dataset including the Lucene index, topics, and relevance judgments.

    Returns:
        Tuple[LuceneSearcher, TopicDict, QrelsDict]: Searcher, topics, and qrels
    """
    # Initialize Lucene searcher with BioASQ index
    searcher = LuceneSearcher(INDEX_DIR)

    # Load topics (queries)
    with open(QUERIES_FILE, 'r') as f:
        topics = json.load(f)

    # Load qrels (relevance judgments)
    qrels = {}
    with open(QRELS_FILE, 'r') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(relevance)

    return searcher, topics, qrels


def split_dataset(topics: TopicDict, qrels: QrelsDict, test_size: float = 0.2, val_size: float = 0.2) -> Tuple[Dict[str, TopicDict], Dict[str, QrelsDict]]:
    """
    Split the dataset into training, validation, and test sets.

    Args:
        topics (TopicDict): Dictionary of topics
        qrels (QrelsDict): Dictionary of relevance judgments
        test_size (float): Proportion of the dataset to include in the test split
        val_size (float): Proportion of the training set to include in the validation split

    Returns:
        Tuple[Dict[str, TopicDict], Dict[str, QrelsDict]]: Split topics and qrels
    """
    topic_ids = list(topics.keys())
    train_val_ids, test_ids = train_test_split(topic_ids, test_size=test_size, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size / (1 - test_size), random_state=42)

    split_topics = {
        'train': {tid: topics[tid] for tid in train_ids},
        'val': {tid: topics[tid] for tid in val_ids},
        'test': {tid: topics[tid] for tid in test_ids}
    }
    split_qrels = {
        'train': {tid: qrels[tid] for tid in train_ids},
        'val': {tid: qrels[tid] for tid in val_ids},
        'test': {tid: qrels[tid] for tid in test_ids}
    }

    return split_topics, split_qrels


def average_precision_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int = 10) -> float:
    """
    Calculate the Average Precision at K for a single query.

    Args:
        relevant_docs (Set[str]): Set of relevant document IDs
        retrieved_docs (List[str]): List of retrieved document IDs
        k (int): Number of documents to consider

    Returns:
        float: Average Precision at K
    """
    relevant_retrieved = 0
    sum_precision = 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            relevant_retrieved += 1
            sum_precision += relevant_retrieved / (i + 1)
    return sum_precision / min(len(relevant_docs), k) if relevant_docs else 0


def evaluate_bm25(searcher: LuceneSearcher, topics: TopicDict, qrels: QrelsDict, k1: float, b: float) -> float:
    """
    Evaluate BM25 performance using Mean Average Precision at 10 (MAP@10).

    Args:
        searcher (LuceneSearcher): Initialized Lucene searcher
        topics (TopicDict): Dictionary of topics to evaluate
        qrels (QrelsDict): Dictionary of relevance judgments
        k1 (float): BM25 k1 parameter
        b (float): BM25 b parameter

    Returns:
        float: MAP@10 score
    """
    searcher.set_bm25(k1, b)
    ap_scores = []

    for topic_id, topic in topics.items():
        query = topic['text']
        hits = searcher.search(query, k=10)
        retrieved_docs = [hit.docid for hit in hits]
        relevant_docs = set(qrels[topic_id].keys())
        ap = average_precision_at_k(relevant_docs, retrieved_docs, k=10)
        ap_scores.append(ap)

    return np.mean(ap_scores)


def grid_search(searcher: LuceneSearcher, val_topics: TopicDict, val_qrels: QrelsDict) -> Tuple[float, float, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform grid search to find optimal BM25 parameters.

    Args:
        searcher (LuceneSearcher): Initialized Lucene searcher
        val_topics (TopicDict): Validation set topics
        val_qrels (QrelsDict): Validation set relevance judgments

    Returns:
        Tuple[float, float, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
            Best k1, best b, results DataFrame, grid data, k1 range, b range
    """
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


def save_heatmap_data(grid_data: np.ndarray, k1_range: np.ndarray, b_range: np.ndarray, filename: str = 'bm25_heatmap_data.txt') -> None:
    """
    Save heatmap data to a file for visualization in tools like Origin.

    Args:
        grid_data (np.ndarray): 2D array of MAP@10 scores
        k1_range (np.ndarray): Range of k1 values
        b_range (np.ndarray): Range of b values
        filename (str): Output filename
    """
    with open(filename, 'w') as f:
        f.write("k1\\b\t" + "\t".join(map(str, b_range)) + "\n")
        for i, k1 in enumerate(k1_range):
            row_data = [str(k1)] + [str(val) for val in grid_data[i, :]]
            f.write("\t".join(row_data) + "\n")
    print(f"Heatmap data saved to {filename}")


def main():
    """
    Main execution function to optimize BM25 parameters for BioASQ dataset.
    """
    print("Loading BioASQ dataset...")
    searcher, topics, qrels = load_bioasq_dataset()

    print("Splitting dataset...")
    split_topics, split_qrels = split_dataset(topics, qrels)

    print("Performing grid search...")
    best_k1, best_b, results_df, grid_data, k1_range, b_range = grid_search(
        searcher, split_topics['val'], split_qrels['val']
    )

    print(f"Best parameters: k1={best_k1}, b={best_b}")

    print("Evaluating on test set...")
    test_map = evaluate_bm25(searcher, split_topics['test'], split_qrels['test'], best_k1, best_b)
    print(f"Test set MAP@10: {test_map}")

    # Save results
    results_df.to_csv('bioasq_bm25_grid_search_results.csv', index=False)
    with open('bioasq_best_params.json', 'w') as f:
        json.dump({'k1': best_k1, 'b': best_b, 'test_map@10': test_map}, f)

    # Save heatmap data for visualization
    save_heatmap_data(grid_data, k1_range, b_range)


if __name__ == '__main__':
    main()
