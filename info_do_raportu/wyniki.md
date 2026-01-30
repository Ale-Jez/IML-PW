on 10 epochs



======================================================================
🧪 RUNNING MODEL EVALUATION - full version 
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9801 (98.01%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9715    0.9898    0.9805     12654
Group Member (1)     0.9894    0.9703    0.9797     12370

        accuracy                         0.9801     25024
       macro avg     0.9804    0.9800    0.9801     25024
    weighted avg     0.9803    0.9801    0.9801     25024


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [12525    129]
  Actual In   [  368  12002]

🔐 Access Control Metrics:
   True Positives (correctly granted):  12002
   True Negatives (correctly denied):   12525
   False Positives (wrongly granted):   129
   False Negatives (wrongly denied):    368

   False Acceptance Rate (FAR): 0.0102 (1.02%)
   False Rejection Rate (FRR): 0.0297 (2.97%)
   Equal Error Rate (EER) ≈: 0.0200


======================================================================
🧪 RUNNING MODEL EVALUATION -------- SGD instead of AdamW
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9594 (95.94%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9694    0.9498    0.9595     12654
Group Member (1)     0.9497    0.9693    0.9594     12370

        accuracy                         0.9594     25024
       macro avg     0.9595    0.9595    0.9594     25024
    weighted avg     0.9596    0.9594    0.9594     25024


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [12019    635]
  Actual In   [  380  11990]

🔐 Access Control Metrics:
   True Positives (correctly granted):  11990
   True Negatives (correctly denied):   12019
   False Positives (wrongly granted):   635
   False Negatives (wrongly denied):    380

   False Acceptance Rate (FAR): 0.0502 (5.02%)
   False Rejection Rate (FRR): 0.0307 (3.07%)
   Equal Error Rate (EER) ≈: 0.0405
======================================================================


📊 Label distribution (binary):
   0 (outsider):     25277 chunks
   1 (group member): 4387 chunks

Total chunks: 29664

======================================================================
🧪 RUNNING MODEL EVALUATION  ------- NO AUGMENTATION
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9625 (96.25%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9731    0.9833    0.9782      2575
Group Member (1)     0.8959    0.8409    0.8675       440

        accuracy                         0.9625      3015
       macro avg     0.9345    0.9121    0.9228      3015
    weighted avg     0.9618    0.9625    0.9620      3015


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [ 2532     43]
  Actual In   [   70    370]

🔐 Access Control Metrics:
   True Positives (correctly granted):  370
   True Negatives (correctly denied):   2532
   False Positives (wrongly granted):   43
   False Negatives (wrongly denied):    70

   False Acceptance Rate (FAR): 0.0167 (1.67%)
   False Rejection Rate (FRR): 0.1591 (15.91%)
   Equal Error Rate (EER) ≈: 0.0879
======================================================================


======================================================================
🧪 RUNNING MODEL EVALUATION ----------- NO AUG 100 epochs
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9884 (98.84%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9915    0.9950    0.9932      2575
Group Member (1)     0.9698    0.9500    0.9598       440

        accuracy                         0.9884      3015
       macro avg     0.9807    0.9725    0.9765      3015
    weighted avg     0.9883    0.9884    0.9883      3015


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [ 2562     13]
  Actual In   [   22    418]

🔐 Access Control Metrics:
   True Positives (correctly granted):  418
   True Negatives (correctly denied):   2562
   False Positives (wrongly granted):   13
   False Negatives (wrongly denied):    22

   False Acceptance Rate (FAR): 0.0050 (0.50%)
   False Rejection Rate (FRR): 0.0500 (5.00%)
   Equal Error Rate (EER) ≈: 0.0275
======================================================================


=====================
OPTIMIZER COMPARISION
=====================


======================================================================
🧪 RUNNING MODEL EVALUATION            SGD+Cosine
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9615 (96.15%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9526    0.9722    0.9623     12654
Group Member (1)     0.9709    0.9505    0.9606     12370

        accuracy                         0.9615     25024
       macro avg     0.9618    0.9614    0.9615     25024
    weighted avg     0.9617    0.9615    0.9615     25024


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [12302    352]
  Actual In   [  612  11758]

🔐 Access Control Metrics:
   True Positives (correctly granted):  11758
   True Negatives (correctly denied):   12302
   False Positives (wrongly granted):   352
   False Negatives (wrongly denied):    612

   False Acceptance Rate (FAR): 0.0278 (2.78%)
   False Rejection Rate (FRR): 0.0495 (4.95%)
   Equal Error Rate (EER) ≈: 0.0386
======================================================================





======================================================================
🧪 RUNNING MODEL EVALUATION --------------  AdamW + Cosine
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9801 (98.01%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9871    0.9734    0.9802     12654
Group Member (1)     0.9731    0.9870    0.9800     12370

        accuracy                         0.9801     25024
       macro avg     0.9801    0.9802    0.9801     25024
    weighted avg     0.9802    0.9801    0.9801     25024


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [12317    337]
  Actual In   [  161  12209]

🔐 Access Control Metrics:
   True Positives (correctly granted):  12209
   True Negatives (correctly denied):   12317
   False Positives (wrongly granted):   337
   False Negatives (wrongly denied):    161

   False Acceptance Rate (FAR): 0.0266 (2.66%)
   False Rejection Rate (FRR): 0.0130 (1.30%)
   Equal Error Rate (EER) ≈: 0.0198
======================================================================



======================================================================
🧪 RUNNING MODEL EVALUATION --------------- AdamW + OneCycleLR
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9399 (93.99%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9340    0.9481    0.9410     12654
Group Member (1)     0.9461    0.9314    0.9387     12370

        accuracy                         0.9399     25024
       macro avg     0.9400    0.9398    0.9398     25024
    weighted avg     0.9399    0.9399    0.9398     25024


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [11997    657]
  Actual In   [  848  11522]

🔐 Access Control Metrics:
   True Positives (correctly granted):  11522
   True Negatives (correctly denied):   11997
   False Positives (wrongly granted):   657
   False Negatives (wrongly denied):    848

   False Acceptance Rate (FAR): 0.0519 (5.19%)
   False Rejection Rate (FRR): 0.0686 (6.86%)
   Equal Error Rate (EER) ≈: 0.0602
======================================================================






======================================================================
🧪 RUNNING MODEL EVALUATION    ----------------SGD + OneCycleLR
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.8888 (88.88%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.8676    0.9207    0.8933     12654
Group Member (1)     0.9134    0.8563    0.8839     12370

        accuracy                         0.8888     25024
       macro avg     0.8905    0.8885    0.8886     25024
    weighted avg     0.8902    0.8888    0.8887     25024


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [11650   1004]
  Actual In   [ 1778  10592]

🔐 Access Control Metrics:
   True Positives (correctly granted):  10592
   True Negatives (correctly denied):   11650
   False Positives (wrongly granted):   1004
   False Negatives (wrongly denied):    1778

   False Acceptance Rate (FAR): 0.0793 (7.93%)
   False Rejection Rate (FRR): 0.1437 (14.37%)
   Equal Error Rate (EER) ≈: 0.1115
======================================================================

=====================
WEIGHT INITIALIZATION - AdamW + Cosine
=====================




======================================================================
🧪 RUNNING MODEL EVALUATION --------------- Default(Xavier)
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9399 (93.99%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9340    0.9481    0.9410     12654
Group Member (1)     0.9461    0.9314    0.9387     12370

        accuracy                         0.9399     25024
       macro avg     0.9400    0.9398    0.9398     25024
    weighted avg     0.9399    0.9399    0.9398     25024


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [11997    657]
  Actual In   [  848  11522]

🔐 Access Control Metrics:
   True Positives (correctly granted):  11522
   True Negatives (correctly denied):   11997
   False Positives (wrongly granted):   657
   False Negatives (wrongly denied):    848

   False Acceptance Rate (FAR): 0.0519 (5.19%)
   False Rejection Rate (FRR): 0.0686 (6.86%)
   Equal Error Rate (EER) ≈: 0.0602
======================================================================




Kaiming

======================================================================
🧪 RUNNING MODEL EVALUATION  --------- Kaiming
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9821 (98.21%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9891    0.9753    0.9822     12654
Group Member (1)     0.9751    0.9890    0.9820     12370

        accuracy                         0.9821     25024
       macro avg     0.9821    0.9822    0.9821     25024
    weighted avg     0.9822    0.9821    0.9821     25024


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [12342    312]
  Actual In   [  136  12234]

🔐 Access Control Metrics:
   True Positives (correctly granted):  12234
   True Negatives (correctly denied):   12342
   False Positives (wrongly granted):   312
   False Negatives (wrongly denied):    136

   False Acceptance Rate (FAR): 0.0247 (2.47%)
   False Rejection Rate (FRR): 0.0110 (1.10%)
   Equal Error Rate (EER) ≈: 0.0178
======================================================================



======================================================================
🧪 RUNNING MODEL EVALUATION --------------- Orthogonal
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9821 (98.21%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9853    0.9793    0.9823     12654
Group Member (1)     0.9790    0.9850    0.9820     12370

        accuracy                         0.9821     25024
       macro avg     0.9821    0.9822    0.9821     25024
    weighted avg     0.9822    0.9821    0.9821     25024


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [12392    262]
  Actual In   [  185  12185]

🔐 Access Control Metrics:
   True Positives (correctly granted):  12185
   True Negatives (correctly denied):   12392
   False Positives (wrongly granted):   262
   False Negatives (wrongly denied):    185

   False Acceptance Rate (FAR): 0.0207 (2.07%)
   False Rejection Rate (FRR): 0.0150 (1.50%)
   Equal Error Rate (EER) ≈: 0.0178
======================================================================


===================
MC Dropout
===================
🎲 Starting Monte Carlo Dropout Test
   MC Samples: 30
   Max Batches: 20

🔬 Running MC Dropout Evaluation (n_samples=30)...
  Processed 10 batches...
  Processed 20 batches...

======================================================================
📊 MONTE CARLO DROPOUT RESULTS
======================================================================
Accuracy: 44.49%

🟢 CORRECT PREDICTIONS:
   Embedding Uncertainty: 0.03023 ± 0.00048
   Mean Confidence:       0.51019
   Agreement Score:       0.68161

🔴 INCORRECT PREDICTIONS:
   Embedding Uncertainty: 0.03016 ± 0.00051
   Mean Confidence:       0.51073
   Agreement Score:       0.69663

📈 Uncertainty Ratio (Incorrect/Correct): 0.998x
   ❌ Poor separation: Correct predictions have higher uncertainty

📊 Agreement Difference (Correct - Incorrect): -0.015
   ❌ Poor: Little agreement difference
======================================================================

======================================================================
💡 PRACTICAL RECOMMENDATIONS
======================================================================

📊 EMBEDDING UNCERTAINTY (keep if ≤ threshold):
   Threshold: 0.03091
   Accuracy:  44.37%
   Coverage:  93.0%

🤝 AGREEMENT SCORE (keep if ≥ threshold):
   Threshold: 0.50000
   Accuracy:  44.49%
   Coverage:  100.0%

======================================================================
✅ Threshold analysis complete!
