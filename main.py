import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_dataset,predict
from datetime import datetime
from fastapi.staticfiles import StaticFiles
##from fastapi.encoders import jsonable_encoder
##from fastapi.responses import JSONResponse
##from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI( title="CredScore Application",docs_url="/")
##app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_event_handler("startup", load_dataset)

class QueryIn(BaseModel):
    existing_checking_account :str
    duration_in_month :int
    credit_history :str
    purpose :str
    credit_amount :int
    savings_account_bonds :str
    present_employment_since :str
    percentage_of_disposable_income :int
    personal_status_and_sex :str
    other_debtors_guarantors :str
    present_residence_since :int
    property :str
    age_in_years :int
    other_installment_plans :str
    housing :str
    number_of_existing_credits_at_this_bank :int
    job :str
    number_of_people_being_liable :int
    telephone :str
    foreign_worker :str

class QueryOut(BaseModel):
    is_trusted_customer: str
    timestamp_str:str

@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict_customer", response_model=QueryOut, status_code=200)
async def predict_customer( query_data: QueryIn):
    print(f"query_data={query_data}")
    result = predict(query_data)
    ct     = datetime.now()
    ctStr  = ct.strftime("%m/%d/%Y, %H:%M:%S")
    output = {'is_trusted_customer': result,'timestamp_str':ctStr}
    
    print(f"output={output}")

    return output

@app.get("/demo",response_class=FileResponse)
async def demo():
    return FileResponse("static/CredScore_Application.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8088, reload=True)
