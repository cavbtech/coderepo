from fastapi.testclient import TestClient
from main import app
import pytest
from _pytest.terminal import TerminalReporter

@pytest.mark.trylast
def pytest_configure(config):
    vanilla_reporter = config.pluginmanager.getplugin("terminalreporter")
    my_reporter = MyReporter(config)
    config.pluginmanager.unregister(vanilla_reporter)
    config.pluginmanager.register(my_reporter, "terminalreporter")




def test_pred_good_customer_with_textdata():
    payload = {
        "existing_checking_account":"A13",
        "duration_in_month":18,
        "credit_history":"A32",
        "purpose":"A43",
        "credit_amount":2100,
        "savings_account_bonds":"A61",
        "present_employment_since":"A73",
        "percentage_of_disposable_income":4,
        "personal_status_and_sex":"A93",
        "other_debtors_guarantors":"A102",
        "present_residence_since":2,
        "property":"A121",
        "age_in_years":37,
        "other_installment_plans":"A142",
        "housing":"A152",
        "number_of_existing_credits_at_this_bank":1,
        "job":"A173",
        "number_of_people_being_liable":1,
        "telephone":"A191",
        "foreign_worker":"A201"
    }
    with TestClient(app) as client:
        response = client.post('/predict_customer', json=payload)
        flower_result = response.json()['is_trusted_customer']
        assert response.status_code == 200
        print(f"response.json()={response.json()}")
        assert flower_result == "Good risk"