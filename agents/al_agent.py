"""
Active Learning Agent — strategy_a dataset
Target: label_binary (liked / not_liked)
Features: TF-IDF on review_text
Strategies: entropy sampling vs random baseline
"""

import io
import sys
import json
import numpy as np

# Fix Windows console encoding only when running as a script (not in Jupyter)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import entropy as scipy_entropy

# ─── Config ───────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
DATA_PATH      = Path(__file__).parent.parent / "data/labeled/strategy_a_labeled.csv"
REPORT_PATH    = Path(__file__).parent.parent / "data/reports/al_results.json"
TARGET_COL     = "label_binary"
TEXT_COL       = "review_text"
N_INIT         = 50
N_ITERATIONS   = 5
BATCH_SIZE     = 10
TEST_SIZE      = 0.20
STRATEGIES     = ["entropy", "random"]


# ─── Data loading & splitting ─────────────────────────────────────────────────
def load_and_split(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    df = df[[TEXT_COL, TARGET_COL]].dropna().reset_index(drop=True)

    # 1. Hold-out test set (20%, stratified, fixed)
    idx_trainpool, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=df[TARGET_COL],
        random_state=RANDOM_STATE
    )

    # 2. Initial labeled (N_INIT from trainpool, stratified)
    df_trainpool = df.loc[idx_trainpool].reset_index(drop=True)
    idx_init, idx_pool = train_test_split(
        df_trainpool.index, train_size=N_INIT,
        stratify=df_trainpool[TARGET_COL],
        random_state=RANDOM_STATE
    )

    test_df    = df.loc[idx_test].reset_index(drop=True)
    init_df    = df_trainpool.loc[idx_init].reset_index(drop=True)
    pool_df    = df_trainpool.loc[idx_pool].reset_index(drop=True)

    return init_df, pool_df, test_df


# ─── Feature extraction ───────────────────────────────────────────────────────
def build_vectorizer(texts):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                          sublinear_tf=True)
    vec.fit(texts)
    return vec


# ─── Sampling strategies ──────────────────────────────────────────────────────
def query_entropy(model, vec, pool_df, batch_size):
    X_pool = vec.transform(pool_df[TEXT_COL])
    proba  = model.predict_proba(X_pool)
    ent    = scipy_entropy(proba, axis=1)          # higher = more uncertain
    top_idx = np.argsort(ent)[::-1][:batch_size]
    return top_idx


def query_random(pool_df, batch_size, rng):
    return rng.choice(len(pool_df), size=batch_size, replace=False)


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model, vec, df):
    X = vec.transform(df[TEXT_COL])
    y = df[TARGET_COL]
    preds = model.predict(X)
    return {
        "accuracy": round(accuracy_score(y, preds), 4),
        "f1":       round(f1_score(y, preds, average="macro"), 4),
    }


# ─── Single AL cycle ──────────────────────────────────────────────────────────
def run_al_cycle(strategy, init_df, pool_df, test_df, verbose=True):
    labeled_df = init_df.copy()
    pool_df    = pool_df.copy()
    rng        = np.random.default_rng(RANDOM_STATE)

    # Fit vectorizer on all available text (labeled + pool) — realistic scenario
    all_texts = pd.concat([labeled_df[TEXT_COL], pool_df[TEXT_COL]])
    vec = build_vectorizer(all_texts)

    results = []

    for iteration in range(N_ITERATIONS + 1):
        # Train
        X_train = vec.transform(labeled_df[TEXT_COL])
        y_train = labeled_df[TARGET_COL]
        model   = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
        model.fit(X_train, y_train)

        # Evaluate on test
        metrics = evaluate(model, vec, test_df)
        results.append({
            "iteration":    iteration,
            "n_labeled":    len(labeled_df),
            "accuracy":     metrics["accuracy"],
            "f1":           metrics["f1"],
        })

        if verbose:
            tag = f"Iteration {iteration}:"
            action = f"fit на labeled ({len(labeled_df)})"
            print(f"  {tag} {action} → accuracy={metrics['accuracy']}, F1={metrics['f1']}")

        # Query next batch (skip on last iteration)
        if iteration < N_ITERATIONS and len(pool_df) >= BATCH_SIZE:
            if strategy == "entropy":
                query_idx = query_entropy(model, vec, pool_df, BATCH_SIZE)
            else:
                query_idx = query_random(pool_df, BATCH_SIZE, rng)

            new_samples = pool_df.iloc[query_idx]
            labeled_df  = pd.concat([labeled_df, new_samples], ignore_index=True)
            pool_df     = pool_df.drop(pool_df.index[query_idx]).reset_index(drop=True)

    return results, model, vec


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Active Learning Experiment")
    print("  Dataset:    strategy_a  |  Target: label_binary")
    print(f"  N_init={N_INIT}, iterations={N_ITERATIONS}, batch={BATCH_SIZE}")
    print("=" * 55)

    init_df, pool_df, test_df = load_and_split()

    print(f"\n📊 Data split:")
    print(f"   Test:           {len(test_df)} samples")
    print(f"   Initial labeled:{len(init_df)} samples")
    print(f"   Unlabeled pool: {len(pool_df)} samples")
    print(f"   Class dist (init): {dict(init_df[TARGET_COL].value_counts())}")

    all_results = {}

    for strategy in STRATEGIES:
        print(f"\n{'═'*55}")
        print(f"  Strategy: {strategy.upper()}")
        print(f"{'═'*55}")
        results, model, vec = run_al_cycle(
            strategy, init_df, pool_df, test_df, verbose=True
        )
        all_results[strategy] = results

        init_f1  = results[0]["f1"]
        final_f1 = results[-1]["f1"]
        print(f"\n  ✓ Final accuracy={results[-1]['accuracy']}, F1={final_f1}")
        print(f"  ✓ Improvement: {init_f1} → {final_f1} (+{round(final_f1 - init_f1, 4)})")

    # ── Comparison ──
    print(f"\n{'═'*55}")
    print("  COMPARISON")
    print(f"{'═'*55}")
    print(f"{'Iter':>5} {'n_labeled':>10}  {'entropy acc':>12} {'entropy F1':>10}  {'random acc':>11} {'random F1':>9}")
    ent_res = all_results["entropy"]
    rnd_res = all_results["random"]
    for i in range(len(ent_res)):
        print(
            f"  {i:>3}  {ent_res[i]['n_labeled']:>9}   "
            f"{ent_res[i]['accuracy']:>11}  {ent_res[i]['f1']:>9}   "
            f"{rnd_res[i]['accuracy']:>10}  {rnd_res[i]['f1']:>8}"
        )

    # ── Savings ──
    ent_final_f1 = ent_res[-1]["f1"]
    rnd_final_f1 = rnd_res[-1]["f1"]
    rnd_f1_values = [r["f1"] for r in rnd_res]
    ent_f1_values = [r["f1"] for r in ent_res]

    # Find how many samples entropy needs to match random's final F1
    savings_samples = None
    for r in ent_res:
        if r["f1"] >= rnd_final_f1:
            savings_samples = r["n_labeled"]
            break

    rnd_final_n = rnd_res[-1]["n_labeled"]

    print(f"\n💰 Savings:")
    if savings_samples:
        pct = round((1 - savings_samples / rnd_final_n) * 100, 1)
        print(f"   Entropy reaches Random's final F1={rnd_final_f1} at {savings_samples} samples")
        print(f"   vs Random needing {rnd_final_n} samples ({pct}% savings)")
    else:
        print(f"   Entropy F1={ent_final_f1}  vs  Random F1={rnd_final_f1}")
        diff = round(ent_final_f1 - rnd_final_f1, 4)
        print(f"   At equal budget ({rnd_final_n} samples): Entropy {'beats' if diff>0 else 'trails'} Random by {abs(diff)}")

    # ── Save results ──
    report = {
        "config": {
            "dataset": str(DATA_PATH),
            "target": TARGET_COL,
            "n_init": N_INIT,
            "n_iterations": N_ITERATIONS,
            "batch_size": BATCH_SIZE,
            "random_state": RANDOM_STATE,
            "test_size": len(test_df),
        },
        "results": all_results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
