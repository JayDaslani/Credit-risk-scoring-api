# ============================================
# HOME CREDIT RISK — FastAPI
# ============================================
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse,FileResponse
from pydantic import BaseModel,Field, ValidationError
from typing import Literal
import pickle
import json
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App initialize
app = FastAPI(
    title="Home Credit Risk API",
    description="Loan Default Prediction API",
    version="1.0.0"
)

# CORS — frontend se connect hone ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model aur features load karo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

logger.info(f"✅ Model loaded!")
logger.info(f"✅ Features: {len(feature_names)}")


# ============================================
# Input Schema — Pydantic
# ============================================
# Sirf important features lenge user se
# Baaki default values se fill karenge

class LoanApplication(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age must be between 18 and 100")
    income: float = Field(..., gt=0, description="Income must be positive")
    loan_amount: float = Field(..., gt=0)
    employed_years: float = Field(..., ge=0, le=50)
    education: Literal['Higher education', 'Secondary', 'Incomplete higher', 'Lower secondary', 'Academic degree']
    gender: Literal['M', 'F'] = Field(..., description="Gender must be M or F")
    own_car: Literal['Y', 'N']
    own_realty: Literal['Y', 'N']
    ext_source_1: float = Field(0.5, ge=0, le=1)
    ext_source_2: float = Field(0.5, ge=0, le=1)
    ext_source_3: float = Field(0.5, ge=0, le=1)


@app.exception_handler(ValidationError)
async def validation_exception_handler(
    request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "status" : "error",
            "message": "Invalid input data",
            "details": exc.errors()
        }
    )
# ============================================
# Helper Function — Input Prepare karo
# ============================================
def prepare_input(data: LoanApplication) -> pd.DataFrame:
    # Ek row ka DataFrame banao
    # Saari features default 0 se start
    input_dict = {col: 0 for col in feature_names}

    # User ki values fill karo
    input_dict['AGE_YEARS'] = data.age
    input_dict['EMPLOYED_YEARS'] = data.employed_years
    input_dict['AMT_CREDIT'] = data.loan_amount
    input_dict['EXT_SOURCE_1'] = data.ext_source_1
    input_dict['EXT_SOURCE_2'] = data.ext_source_2
    input_dict['EXT_SOURCE_3'] = data.ext_source_3

    # Engineered features calculate karo
    input_dict['EXT_SOURCE_MEAN'] = np.mean([
        data.ext_source_1,
        data.ext_source_2,
        data.ext_source_3
    ])
    input_dict['EXT_SOURCE_MIN'] = min([
        data.ext_source_1,
        data.ext_source_2,
        data.ext_source_3
    ])
    input_dict['EXT_SOURCE_STD'] = np.std([
        data.ext_source_1,
        data.ext_source_2,
        data.ext_source_3
    ])
    input_dict['EXT_MEAN_X_AGE'] = (
            input_dict['EXT_SOURCE_MEAN'] * data.age
    )
    input_dict['EXT_MEAN_X_EMPLOY'] = (
            input_dict['EXT_SOURCE_MEAN'] *
            data.employed_years
    )

    # Categorical features
    if data.gender == 'M':
        input_dict['CODE_GENDER_M'] = 1
    if data.own_car == 'Y':
        input_dict['FLAG_OWN_CAR_Y'] = 1
    if data.own_realty == 'Y':
        input_dict['FLAG_OWN_REALTY_Y'] = 1

    # Education
    edu_map = {
        'Higher education':
            'NAME_EDUCATION_TYPE_Higher_education',
        'Secondary':
            'NAME_EDUCATION_TYPE_Secondary / secondary special',
        'Incomplete higher':
            'NAME_EDUCATION_TYPE_Incomplete_higher',
        'Lower secondary':
            'NAME_EDUCATION_TYPE_Lower_secondary',
    }
    if data.education in edu_map:
        col = edu_map[data.education]
        if col in input_dict:
            input_dict[col] = 1

    return pd.DataFrame([input_dict])


# ============================================
# Routes
# ============================================
@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
        "version": "1.0.0"
    }

def business_validation(data):
    # Loan income se 20x zyada nahi hona chahiye
    if data.loan_amount > data.income * 20:
        raise HTTPException(
            status_code=400,
            detail="Loan amount income se "
                   "20x zyada nahi ho sakta!"
        )
    # Employment age se zyada nahi hona chahiye
    if data.employed_years > data.age - 16:
        raise HTTPException(
            status_code=400,
            detail="Employment years "
                   "age se zyada nahi ho sakte!"
        )

@app.post("/predict")
def predict(application: LoanApplication):
    business_validation(application)
    logger.info(
        f"Prediction request — "
        f"age: {application.age}, "
        f"income: {application.income}"
    )
    try:
        # 1. Input prepare karne ki koshish karo
        input_df = prepare_input(application)

        # 2. Model se prediction lo
        # Check karo ki kya model load hua hai
        if model is None:
            raise ValueError("Machine Learning Model is not loaded on the server.")

        probability = model.predict_proba(input_df)[0][1]

        # 3. Risk logic (Aapka purana logic)
        if probability < 0.3:
            risk_level, decision, color = "LOW RISK", "APPROVED ✅", "green"
        elif probability < 0.6:
            risk_level, decision, color = "MEDIUM RISK", "REVIEW NEEDED ⚠️", "yellow"
        else:
            risk_level, decision, color = "HIGH RISK", "REJECTED ❌", "red"

        return {
            "status": "success",
            "probability": round(float(probability), 4),
            "risk_percent": f"{probability * 100:.1f}%",
            "risk_level": risk_level,
            "decision": decision,
            "color": color,
            "input_summary": {
                "income": application.income,
                "loan_amount": application.loan_amount
            }
        }

    except ValueError as ve:
        # Business logic ya missing model ke errors
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(ve)}
        )

    except Exception as e:
        print(f"ERROR: Something went wrong: {str(e)}")
        # Koi bhi unknown error (jaise data mismatch)
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "An unexpected error occurred during prediction.",
                "details": str(e)
            }
        )
