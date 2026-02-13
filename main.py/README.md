# ML Classification Models - Assignment 2

## a. Problem Statement

This project compares 6 different classification algorithms to predict loan default risk using a loan approval dataset. The goal is to train multiple ML models and find out which one performs best for this problem. All models are evaluated using 6 different metrics - Accuracy, AUC, Precision, Recall, F1 Score, and MCC. A Streamlit web application is also included where users can upload test data and see how each model performs.

## b. Dataset Description

The Loan Approval Dataset from Kaggle is used for this assignment. It's a binary classification problem where the task is to predict whether a loan will be approved or rejected/defaulted.

**Dataset Details:**
- Total rows: 45,000
- Total features: 13 (12 input features + 1 target)
- Target variable: loan_status (0 = Approved, 1 = Rejected/Default)
- Data types: Mix of numerical and categorical
- Class imbalance: The dataset is imbalanced with 77.8% Class 0 and only 22.2% Class 1

**List of Features:**
1. person_age - Age of the person applying for loan
2. person_income - Annual income
3. person_home_ownership - Whether they rent, own, have mortgage, or other
4. person_emp_exp - Years of work experience
5. loan_intent - Purpose of the loan (personal, education, medical, business, etc.)
6. loan_grade - Credit grade from A to G
7. loan_amnt - Loan amount requested
8. loan_int_rate - Interest rate on the loan
9. loan_percent_income - Loan amount as a percentage of income
10. cb_person_default_on_file - Whether they defaulted before (Y or N)
11. cb_person_cred_hist_length - Length of credit history in years
12. loan_status - Target variable (0 = Approved, 1 = Rejected/Default)

## c. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8970 | 0.9544 | 0.7762 | 0.7563 | 0.7661 | 0.7002 |
| Decision Tree | 0.8830 | 0.9650 | 0.6766 | 0.9111 | 0.7766 | 0.7139 |
| kNN | 0.8912 | 0.9283 | 0.7811 | 0.7121 | 0.7450 | 0.6773 |
| Naive Bayes | 0.7218 | 0.9401 | 0.4450 | 1.0000 | 0.6159 | 0.5344 |
| Random Forest (Ensemble) | 0.9058 | 0.9723 | 0.7453 | 0.8781 | 0.8063 | 0.7490 |
| XGBoost (Ensemble) | 0.9307 | 0.9764 | 0.8929 | 0.7831 | 0.8344 | 0.7935 |

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Provides a decent baseline with F1 of 0.7661 and MCC of 0.7002. The precision (0.7762) and recall (0.7563) are pretty balanced. Easy to interpret because of the linear decision boundary. The AUC is quite good at 0.9544. However, being linear means it can't capture complex patterns in the data. |
| Decision Tree | Achieves F1 of 0.7766 and MCC of 0.7139. The recall is really high (0.9111) which means it catches most of the loan defaults, but precision is lower (0.6766) so there are more false alarms. AUC is strong at 0.9650. Tends to overfit without proper pruning. |
| kNN | Shows F1 of 0.7450 and MCC of 0.6773. Has the best precision (0.7811) among the non-ensemble models but misses some defaults because recall is lower (0.7121). Choosing the right k value matters a lot. Also, predictions are slower compared to other models. |
| Naive Bayes | Performs the worst with F1 of 0.6159 and MCC of 0.5344. Catches all the defaults (recall = 1.0) but has way too many false positives (precision = 0.4450). The AUC is still decent at 0.9401 though. Would only be useful if missing a default is really costly and false alarms are acceptable. |
| Random Forest (Ensemble) | This ensemble method works really well - F1 of 0.8063 and MCC of 0.7490, second best overall. Good balance between precision (0.7453) and recall (0.8781). Handles the class imbalance better than single decision trees and doesn't overfit as much. |
| XGBoost (Ensemble) | Best performing model overall with highest F1 (0.8344) and MCC (0.7935). The precision is excellent at 0.8929 with decent recall (0.7831). Best AUC too at 0.9764. Uses scale_pos_weight parameter to handle the class imbalance effectively. Recommended for deployment. |

