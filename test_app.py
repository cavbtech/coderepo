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

def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"ping":"pong"}

def test_pred_good_customer():
    payload = {
        "existing_checking_account":1,
        "duration_in_month":6,
        "credit_history":4,
        "purpose":12,
        "credit_amount":5,
        "savings_account_bonds":5,
        "present_employment_since":3,
        "percentage_of_disposable_income":4,
        "personal_status_and_sex":1,
        "other_debtors_guarantors":67,
        "present_residence_since":3,
        "property":2,
        "age_in_years":1,
        "other_installment_plans":2,
        "housing":1,
        "number_of_existing_credits_at_this_bank":0,
        "job":0,
        "number_of_people_being_liable":1,
        "telephone":0,
        "foreign_worker":0
    }
    with TestClient(app) as client:
        response = client.post('/predict_customer', json=payload)
        flower_result = response.json()['is_trusted_customer']
        assert response.status_code == 200
        print(f"response.json()={response.json()}")
        assert flower_result == "Good risk"