# AI-ML-Today

End-to-end Titanic survival classification using scikit-learn pipelines.

## Prerequisites

- Python 3.x
- pandas
- numpy
- scikit-learn

## Setup

Activate the existing virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

## Usage

```powershell
python AI_Day.py
```

## What It Demonstrates

- Data loading & exploration
- Feature scaling (StandardScaler)
- One-hot encoding (Sex, Embarked)
- Handling missing values (SimpleImputer)
- Logistic Regression pipeline
- k-NN with scaling vs. unscaled comparison
- Hyperparameter sweep (k values)
- Evaluation metrics (accuracy, precision, recall)

## Expected Output

- Dataset shape: `(800, 9)`
- Train means after scaling: approx. `[0, 0]`
- Train SDs after scaling: approx. `[1, 1]`
- Accuracy with Sex feature
- Accuracy without Sex feature
- `test accuracy` (full ColumnTransformer pipeline)
- `k-NN scaled` accuracy
- `k-NN UNSCALED` accuracy
- Precision and recall at thresholds `0.3`, `0.5`, `0.7`
- `k = 1`, `k = 5`, `k = 15`, `k = 50` accuracies

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
