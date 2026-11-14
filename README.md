# Phase Prediction of High-Entropy Alloys with K-Nearest Neighbors

A machine learning project that predicts the phase of a high-entropy alloy (HEA)
from its chemical composition only, using a K-Nearest Neighbors (KNN) classifier.

## Overview

- **Task:** multi-class classification of the phase of an alloy
  (`BCC`, `FCC`, `BCC+FCC`, `IM` = intermetallic, `AM` = amorphous)
- **Input (features):** the fraction of each element in the alloy (43 element columns,
  fractions sum to 1), computed from the alloy formula with `pymatgen`
- **Output (target):** the phase of the alloy
- **Data:** 3190 alloys, combined from two sources (see Data)
- **Model:** KNN (`scikit-learn`), k = 3, Manhattan distance (p = 1)
- **Result:** 93.9% accuracy on the held-out test set (599 of 638 alloys)

## Workflow

The notebook `HEA_Phase_Prediction_KNN.ipynb` follows these steps:

1. **Data overview and cleaning:** shape, phase classes, missing values, duplicates
2. **Data visualization:** phase distribution, number of elements, number of elements per phase
3. **Feature engineering:** alloy formula to element fractions (`pymatgen`)
4. **Train/test split:** 80% / 20%, stratified by phase (the classes are imbalanced)
5. **Validation and overfitting check:** training vs. validation accuracy for
   different k, repeated over 15 random splits
6. **Hyperparameter tuning:** `GridSearchCV` with 5-fold cross-validation
   (k and distance type) on the training set only
7. **Feature scaling comparison:** raw fractions vs. `MinMaxScaler` vs.
   `StandardScaler`
8. **Final evaluation:** once on the test set (accuracy, classification report,
   confusion matrix)

## Results

Best parameters: k = 3, p = 1 (5-fold cross-validation accuracy 90.7%).
Scaling did not improve the accuracy, so raw fractions were used.
Test accuracy: **93.9%**.

| Phase | Precision | Recall | F1-score | Test samples |
|---|---|---|---|---|
| AM | 0.97 | 0.99 | 0.98 | 104 |
| BCC | 0.93 | 0.95 | 0.94 | 200 |
| BCC+FCC | 0.88 | 0.86 | 0.87 | 134 |
| FCC | 0.96 | 0.96 | 0.96 | 168 |
| IM | 1.00 | 0.94 | 0.97 | 32 |

AM, FCC and IM are predicted very well. The BCC+FCC class is the hardest
(F1 = 0.87): 34 of the 39 test errors involve this class, mostly confusion
between BCC and BCC+FCC.

## Limitations

- Only the composition is used. Physical descriptors (mixing entropy and enthalpy,
  atomic size mismatch, valence electron concentration) and processing information
  are not included.
- The train/test split is random. Alloys with very similar compositions can appear in
  both sets, so the accuracy may be optimistic for completely new compositions.
  For the same reason, k = 1 was not used, although it gave the highest
  cross-validation accuracy.
- The IM class has only 32 test samples, so its scores are less certain.
- Phase labels from different sources were harmonized to five classes, and this
  involves some judgment.

## Data

The data file is not included in this repository. It was compiled from two
sources and converted to a common format (`Alloy`, `Num_el`, `Phase`):

1. **Own dataset:** collected by the author from the published literature on
   high-entropy alloys.
2. **MPEA dataset:** Borg, C.K.H., Frey, C., Moh, J., Pollock, T.M., Gorsse, S.,
   Miracle, D.B., Senkov, O.N., Meredig, B., Saal, J.E. (2020). Expanded dataset
   of mechanical properties and observed phases of multi-principal element alloys.
   *Scientific Data*, 7, 430. https://doi.org/10.1038/s41597-020-00768-9
   (Figshare: https://doi.org/10.6084/m9.figshare.12642953)

The two sources were merged (3190 alloys in total), duplicates were removed,
and phase labels were harmonized to five classes (BCC, FCC, BCC+FCC, IM, AM).

## How to run

```bash
pip install -r requirements.txt
jupyter notebook HEA_Phase_Prediction_KNN.ipynb
```

The data file is not included. To rerun the notebook, build
`Phase_Prediction_merged.csv` as described in the [Data](#data) section and
place it in the same folder as the notebook (or change the path in the first
code cell).

## Author

Mahnaz Keymasi
