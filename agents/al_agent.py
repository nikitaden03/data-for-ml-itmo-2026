"""
Active Learning Agent: Entropy vs Random sampling.
Standalone script — can be run independently.

Usage:
    python agents/al_agent.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

RANDOM_STATE = 42
TEST_SIZE = 0.20
INITIAL_LABELED = 200
BATCH_SIZE = 100
N_ITERATIONS = 10
DATA_PATH = "data/labeled/strategy_a_labeled.csv"
OUTPUT_PATH = "data/reports/al_results.json"


# ─── Data preparation ─────────────────────────────────────────────────────────

def load_data(path: str):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["text"] = df["body"].fillna("").astype(str)
    X = df["text"].values
    y = df["positive_market_impact"].astype(int).values
    return X, y


def prepare_splits(X, y):
    # Test split (fixed, stratified)
    idx = np.arange(len(X))
    idx_trainpool, idx_test, _, _ = train_test_split(
        idx, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Initial labeled (stratified from trainpool)
    idx_init, idx_pool, _, _ = train_test_split(
        idx_trainpool, y[idx_trainpool],
        train_size=INITIAL_LABELED,
        stratify=y[idx_trainpool],
        random_state=RANDOM_STATE
    )
    return idx_test, idx_init, idx_pool


# ─── Vectorizer ───────────────────────────────────────────────────────────────

def build_vectorizer(X_train):
    vec = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    vec.fit(X_train)
    return vec


# ─── Model ────────────────────────────────────────────────────────────────────

def fit_model(vec, X_labeled, y_labeled):
    X_vec = vec.transform(X_labeled)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    clf.fit(X_vec, y_labeled)
    return clf


def evaluate(vec, clf, X_test, y_test):
    X_vec = vec.transform(X_test)
    y_pred = clf.predict(X_vec)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }


# ─── Query strategies ─────────────────────────────────────────────────────────

def query_entropy(vec, clf, X_pool, batch_size: int) -> np.ndarray:
    """Select batch_size samples with highest entropy."""
    proba = clf.predict_proba(vec.transform(X_pool))
    # Entropy: -sum(p * log(p))
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    return np.argsort(entropy)[-batch_size:]


def query_random(pool_size: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Select batch_size random samples."""
    return rng.choice(pool_size, size=batch_size, replace=False)


# ─── AL loop ──────────────────────────────────────────────────────────────────

def run_al_loop(X, y, idx_test, idx_init, idx_pool, strategy: str, verbose: bool = True):
    X_test, y_test = X[idx_test], y[idx_test]

    labeled = list(idx_init.copy())
    pool = list(idx_pool.copy())
    rng = np.random.default_rng(RANDOM_STATE)

    # Build vectorizer on all non-test data
    all_train_idx = np.concatenate([idx_init, idx_pool])
    vec = build_vectorizer(X[all_train_idx])

    history = []

    for iteration in range(N_ITERATIONS + 1):
        X_labeled = X[labeled]
        y_labeled = y[labeled]

        clf = fit_model(vec, X_labeled, y_labeled)
        metrics = evaluate(vec, clf, X_test, y_test)
        metrics["n_labeled"] = len(labeled)
        metrics["iteration"] = iteration
        history.append(metrics)

        if verbose:
            print(f"  iter {iteration:2d} | labeled={len(labeled):5d} | "
                  f"acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f}")

        if iteration == N_ITERATIONS:
            break

        # Query
        X_pool_arr = X[pool]
        if strategy == "entropy":
            chosen_local = query_entropy(vec, clf, X_pool_arr, BATCH_SIZE)
        else:
            chosen_local = query_random(len(pool), BATCH_SIZE, rng)

        chosen_global = [pool[i] for i in chosen_local]
        labeled.extend(chosen_global)
        pool = [p for p in pool if p not in set(chosen_global)]

    return history, clf, vec


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("Loading data...")
    X, y = load_data(DATA_PATH)
    print(f"  Total: {len(X)} samples | balance: {y.mean():.3f}")

    print("Preparing splits...")
    idx_test, idx_init, idx_pool = prepare_splits(X, y)
    print(f"  Test: {len(idx_test)} | Init labeled: {len(idx_init)} | Pool: {len(idx_pool)}")

    print("\n=== ENTROPY STRATEGY ===")
    entropy_history, entropy_clf, entropy_vec = run_al_loop(
        X, y, idx_test, idx_init, idx_pool, strategy="entropy"
    )

    print("\n=== RANDOM STRATEGY ===")
    random_history, random_clf, random_vec = run_al_loop(
        X, y, idx_test, idx_init, idx_pool, strategy="random"
    )

    # Build results
    results = {
        "config": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "initial_labeled": INITIAL_LABELED,
            "batch_size": BATCH_SIZE,
            "n_iterations": N_ITERATIONS,
            "model": "LogisticRegression",
            "vectorizer": "TF-IDF(max_features=10000, ngram=(1,2))",
        },
        "entropy": entropy_history,
        "random": random_history,
    }

    # Summary
    e_final = entropy_history[-1]
    r_final = random_history[-1]
    print(f"\n{'='*45}")
    print("RESULTS SUMMARY")
    print(f"{'='*45}")
    print(f"Entropy  final F1: {e_final['f1']:.4f} | acc: {e_final['accuracy']:.4f}")
    print(f"Random   final F1: {r_final['f1']:.4f} | acc: {r_final['accuracy']:.4f}")
    print(f"Delta F1: {e_final['f1'] - r_final['f1']:+.4f}")

    # Savings: at what labeled size does entropy match random's final F1?
    random_final_f1 = r_final["f1"]
    entropy_match = next(
        (h for h in entropy_history if h["f1"] >= random_final_f1), None
    )
    if entropy_match:
        savings_n = r_final["n_labeled"] - entropy_match["n_labeled"]
        savings_pct = savings_n / r_final["n_labeled"] * 100
        print(f"Entropy matches Random final F1 at {entropy_match['n_labeled']} labels "
              f"(saves {savings_n} = {savings_pct:.1f}%)")
        results["savings"] = {
            "entropy_match_at": entropy_match["n_labeled"],
            "random_final_at": r_final["n_labeled"],
            "savings_n": savings_n,
            "savings_pct": round(savings_pct, 1)
        }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    main()
