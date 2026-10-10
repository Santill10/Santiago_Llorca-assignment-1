"""Training pipeline: baselines + scikit-learn logistic regression.
(PyTorch manual loop and nn.Module versions are added in the next steps.)"""
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from evaluation import LGD, MARGIN, evaluate
from features import Preprocessor, build_raw_features, load_data, split_by_date

MODELS_DIR = Path("models")
C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_data()
    train, val, test = split_by_date(df)
    prep = Preprocessor().fit(train)  # fitted on TRAIN only
    X_tr, X_val, X_te = (prep.transform(d) for d in (train, val, test))
    y_tr, y_val, y_te = (d["default"].to_numpy() for d in (train, val, test))
    amt_val, amt_te = (d["loan_amnt"].to_numpy() for d in (val, test))

    results = {}

    def run(name, s_val, s_te):
        results[name] = evaluate(y_val, s_val, amt_val, y_te, s_te, amt_te)

    # Baseline 1: approve everyone (constant score)
    run("baseline_approve_all", y_val * 0.0, y_te * 0.0)

    # Baseline 2: FICO rule (lower FICO = riskier, so we negate it)
    fico_val = -build_raw_features(val)["fico_mid"].to_numpy()
    fico_te = -build_raw_features(test)["fico_mid"].to_numpy()
    run("baseline_fico", fico_val, fico_te)

    # scikit-learn logistic regression; C chosen by validation log-loss
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=1000).fit(X_tr, y_tr)
        ll = log_loss(y_val, m.predict_proba(X_val)[:, 1])
        print(f"sklearn C={C:<6} val log-loss={ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, C, m)
    _, best_C, sk = best
    run("sklearn", sk.predict_proba(X_val)[:, 1], sk.predict_proba(X_te)[:, 1])

    # Save artifacts: the app only loads these, it never retrains
    joblib.dump(prep, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(sk, MODELS_DIR / "sklearn_logreg.joblib")
    config = {"best_C": best_C, "lgd": LGD, "margin": MARGIN,
              "feature_names": prep.feature_names, "results": results}
    (MODELS_DIR / "results.json").write_text(json.dumps(config, indent=2))

    print(f"\nbest C = {best_C}")
    print(f"{'model':<24}{'test AUC':>9}{'PR-AUC':>9}{'test cost':>12}{'vs approve-all':>16}")
    for name, r in results.items():
        saving = 1 - r["test_cost"] / r["test_cost_approve_all"]
        print(f"{name:<24}{r['test_auc']:>9.3f}{r['test_pr_auc']:>9.3f}"
              f"{r['test_cost']:>12,.0f}{saving:>15.1%}")


if __name__ == "__main__":
    main()