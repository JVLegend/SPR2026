# SPR 2026 - Mammography Report Classification

## Competition Rules
- **Internet is PROHIBITED** in notebook submissions. All notebooks must run with internet disabled.
- Notebooks must use only what's available on Kaggle as Datasets or Models (pre-uploaded inputs).
- Do NOT download models from HuggingFace or any external source at runtime.
- GPU notebooks use T4 x2 accelerator.

## Project Structure
- `kaggle_notebooks/` - Experiment notebooks for Kaggle submission
- `experiments/results.tsv` - Tracking of all experiment scores

## Best Score
- EXP-24 (Specialist Ensemble): **0.81021** - 7 L1 TF-IDF models → LogReg meta-learner, CPU only

## Key Lessons
- Modifying training data (label cleaning, augmentation, pseudo-labeling) consistently worsens results
- More models != better (7 models beats 9-11 models due to meta-learner overfitting)
- Simple greedy threshold search outperforms complex multi-pass approaches
