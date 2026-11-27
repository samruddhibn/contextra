"""
Train and evaluate the CONTEXTRA contextual fusion model.

    python train_fusion_model.py

Writes models/fusion_model.joblib and models/metrics.json and prints the
results. Quote the numbers THIS script prints on your machine, and always say
they are measured on synthetic scenario data.
"""

import fusion_model as fm


def main():
    print("Training fusion model (takes ~20-60 seconds)...")
    bundle, m = fm.train_and_evaluate(save=True)

    print("\n=== DATA ===")
    print(m["data_note"])
    print("Samples:", m["samples"], "| train:", m["train"], "| test:", m["test"],
          "| seed:", m["seed"])
    print("Class balance:", m["class_balance"])

    print("\n=== HELD-OUT TEST RESULTS (synthetic scenarios) ===")
    header = f"{'Model':52s} {'Acc':>6s} {'MacroF1':>8s} {'HIGH recall':>12s}"
    print(header)
    print("-" * len(header))
    for name, s in m["comparisons"].items():
        print(f"{name:52s} {s['accuracy']:6.3f} {s['macro_f1']:8.3f} "
              f"{s['per_class']['HIGH']['recall']:12.3f}")

    print("\nFusion model confusion matrix (rows = true, cols = predicted; LOW, ELEVATED, HIGH):")
    for label, row in zip(fm.LABELS, m["comparisons"]["Fusion model (driver + context)"]["confusion_matrix"]):
        print(f"  {label:9s}", row)

    print(f"\n5-fold CV macro-F1: {m['fusion_cv_macro_f1_mean']:.3f} +/- {m['fusion_cv_macro_f1_std']:.3f}")
    print(f"Inference latency: {m['inference_latency_ms']:.1f} ms per prediction")
    print("\nTop feature importances:")
    for name, val in m["feature_importance"][:8]:
        print(f"  {name:22s} {val:.3f}")

    print("\nSaved: models/fusion_model.joblib, models/metrics.json")
    print("\nHONEST NOTE: labels come from an expert-knowledge simulator, not real accidents.")
    print("The rule engine scores low here partly because it cannot see yawning,")
    print("head pose or distraction. This shows the value of fusing more signals on")
    print("this data, NOT real-world superiority. Validation on real labelled data is future work.")


if __name__ == "__main__":
    main()
