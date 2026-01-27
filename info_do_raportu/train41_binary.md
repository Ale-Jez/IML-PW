======================================================================
🧪 RUNNING MODEL EVALUATION
======================================================================

📋 Detected label_mode: 'binary'
--------------------------------------------------
🔍 Detected label_mode: binary
   → Running BINARY classification test

======================================================================
BINARY CLASSIFICATION TEST RESULTS
======================================================================

📊 Overall Accuracy: 0.9874 (98.74%)

📋 Classification Report:
                  precision    recall  f1-score   support

    Outsider (0)     0.9873    0.9980    0.9926     17856
Group Member (1)     0.9879    0.9259    0.9559      3092

        accuracy                         0.9874     20948
       macro avg     0.9876    0.9620    0.9743     20948
    weighted avg     0.9874    0.9874    0.9872     20948


🔢 Confusion Matrix:
                  Predicted
                  Out   In
  Actual Out  [17821     35]
  Actual In   [  229   2863]

🔐 Access Control Metrics:
   True Positives (correctly granted):  2863
   True Negatives (correctly denied):   17821
   False Positives (wrongly granted):   35
   False Negatives (wrongly denied):    229

   False Acceptance Rate (FAR): 0.0020 (0.20%)
   False Rejection Rate (FRR): 0.0741 (7.41%)
   Equal Error Rate (EER) ≈: 0.0380
======================================================================