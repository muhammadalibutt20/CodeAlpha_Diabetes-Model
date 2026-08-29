from pipeline import MODEL_NAMES


def test_expected_medical_classifiers_are_available():
    expected = {"Logistic Regression", "SVM", "Random Forest", "XGBoost"}
    assert set(MODEL_NAMES) == expected
