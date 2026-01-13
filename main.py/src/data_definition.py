class DataDefinition:
    numeric_cols = ['person_age',
                    'person_income',
                    'person_emp_exp',
                    'loan_amnt',
                    'loan_int_rate',
                    'loan_percent_income',
                    'cb_person_cred_hist_length',
                    'credit_score']

    categorical_cols = ['person_gender',
                        'person_education',
                        'person_home_ownership',
                        'previous_loan_defaults_on_file',
                        'loan_intent']
    target_col = 'loan_status'