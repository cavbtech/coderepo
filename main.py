import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model,predict
from datetime import datetime

app = FastAPI(
    title="CredScore Application",
    docs_url="/"
)

app.add_event_handler("startup", load_model)

class QueryIn(BaseModel):
    existing_checking_account :int
    duration_in_month :int
    credit_history :int
    purpose :int
    credit_amount :int
    savings_account_bonds :int
    present_employment_since :int
    percentage_of_disposable_income :int
    personal_status_and_sex :int
    other_debtors_guarantors :int
    present_residence_since :int
    property :int
    age_in_years :int
    other_installment_plans :int
    housing :int
    number_of_existing_credits_at_this_bank :int
    job :int
    number_of_people_being_liable :int
    telephone :int
    foreign_worker :int

class QueryOut(BaseModel):
    is_trusted_customer: str
    timestamp_str:str

@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict_customer", response_model=QueryOut, status_code=200)
def predict_flower( query_data: QueryIn):
    result = predict(query_data)
    ct     = datetime.now()
    ctStr  = ct.strftime("%m/%d/%Y, %H:%M:%S")
    output = {'trusted': result,'timestamp_str':ctStr}
    return output




if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8888, reload=True)
