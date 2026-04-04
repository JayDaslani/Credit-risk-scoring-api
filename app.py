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
import shap

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
print("Loading SHAP explainer...")
explainer = shap.TreeExplainer(model)
print("✅ SHAP explainer loaded!")

USER_FEATURES = [
    'AGE_YEARS',
    'EMPLOYED_YEARS',
    'AMT_CREDIT',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'FLAG_OWN_CAR_Y',
    'FLAG_OWN_REALTY_Y',
    'CREDIT_TERM',
]
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
    credit_term: float = Field(36, ge=6, le=60,description="Loan repayment in months")
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
    input_dict['CREDIT_TERM'] = data.credit_term
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
    input_dict['EXT_MEAN_X_CREDIT_TERM'] = (
            input_dict['EXT_SOURCE_MEAN'] *
            (1 / (data.credit_term + 1))
    )

    # Categorical features

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
    loan_to_income = data.loan_amount / data.income

    if loan_to_income > 10:
        raise HTTPException(
            status_code=400,
            detail=f"Loan amount cannot exceed 10x "
                   f"your annual income. "
                   f"Maximum allowed loan: "
                   f"₹{data.income * 10:,.0f}"
        )

    if data.loan_amount < 10000:
        raise HTTPException(
            status_code=400,
            detail="Minimum loan amount is ₹10,000"
        )

    if data.loan_amount > 10000000:
        raise HTTPException(
            status_code=400,
            detail="Maximum loan amount is "
                   "₹1,00,00,000 (1 Crore)"
        )

    if data.employed_years > data.age - 16:
        raise HTTPException(
            status_code=400,
            detail="Employment years cannot "
                   "exceed your working age!"
        )

    if data.income < 50000:
        raise HTTPException(
            status_code=400,
            detail="Minimum annual income "
                   "required is ₹50,000"
        )
def get_risk_factors(
        data: LoanApplication,
        probability: float) -> list:

    factors = []

    # Age
    if data.age < 25:
        factors.append(
            "🔴 Young age group (20-25) — "
            "highest default risk (12.3%)"
        )
    elif data.age < 30:
        factors.append(
            "🟡 Age group 25-30 — "
            "above average risk"
        )

    # Employment
    if data.employed_years < 1:
        factors.append(
            "🔴 Less than 1 year of employment — "
            "unstable income history"
        )
    elif data.employed_years < 3:
        factors.append(
            "🟡 Less than 3 years employed — "
            "limited work history"
        )

    # Loan to income ratio
    ratio = data.loan_amount / data.income
    if ratio > 5:
        factors.append(
            f"🔴 High loan-to-income ratio "
            f"({ratio:.1f}x) — significant burden"
        )
    elif ratio > 3:
        factors.append(
            f"🟡 Moderate loan-to-income ratio "
            f"({ratio:.1f}x)"
        )

    # Credit scores
    # app.py mein
    # avg_ext ko CIBIL mein convert karke dikhao
    avg_ext = (data.ext_source_1 +
               data.ext_source_2 +
               data.ext_source_3) / 3

    # CIBIL equivalent
    cibil_equiv = int(avg_ext * 600 + 300)

    if avg_ext < 0.3:
        factors.append(
            f"🔴 Low CIBIL Score (~{cibil_equiv}) — "
            f"poor credit history"
        )
    elif avg_ext < 0.45:
        factors.append(
            f"🟡 Below average CIBIL Score "
            f"(~{cibil_equiv})"
        )
    elif avg_ext >= 0.6:
        factors.append(
            f"🟢 Good CIBIL Score (~{cibil_equiv}) — "
            f"positive factor"
        )

    # Gender


    # Education
    edu_risk = {
        'Lower secondary':
            "🔴 Lower secondary education — "
            "highest risk group (10.9% default rate)",
        'Secondary':
            "🟡 Secondary education — "
            "above average risk",
        'Incomplete higher':
            "🟡 Incomplete higher education",
        'Higher education':
            "🟢 Higher education — "
            "low risk group (5.4% default rate)",
        'Academic degree':
            "🟢 Academic degree — "
            "lowest risk group (1.8% default rate)",
    }
    if data.education in edu_risk:
        factors.append(edu_risk[data.education])

    # Assets
    if data.own_car == 'N' and data.own_realty == 'N':
        factors.append(
            "🟡 No assets (car/property) — "
            "limited financial stability"
        )

    return factors

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

        # Isse replace karo:
        risk_factors = get_risk_factors(
            application, probability)

        # SHAP Explanation
        shap_values = explainer.shap_values(input_df)

        # NEW FORMAT handle karo
        if isinstance(shap_values, list):
            # Old format — list of arrays
            sv = shap_values[1][0]
        else:
            # New format — single array
            # Last dimension = default class
            if len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]

        # Top 5 features nikalo
        feature_names_list = input_df.columns.tolist()
        shap_dict = dict(zip(feature_names_list, sv))

        # Sort by absolute value
        sorted_shap = sorted(
            [(f, v) for f, v in shap_dict.items()
             if f in USER_FEATURES],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        # Clean format
        shap_explanation = []
        for feat, val in sorted_shap:
            direction = "↑ Increases risk" if val > 0 \
                else "↓ Decreases risk"
            shap_explanation.append({
                "feature": feat,
                "impact": round(float(val), 4),
                "direction": direction,
                "abs_impact": round(abs(float(val)), 4)
            })

        print(f"DEBUG SHAP: {shap_explanation}")
        # Debug prints
        print(f"SHAP values type: {type(shap_values)}")
        print(f"SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else len(shap_values)}")
        print(f"SV type: {type(sv)}")
        print(f"SV shape: {sv.shape if hasattr(sv, 'shape') else len(sv)}")
        print(f"SHAP explanation: {shap_explanation}")

        return {
            "status": "success",
            "probability": round(float(probability), 4),
            "risk_percent": f"{probability * 100:.1f}%",
            "risk_level": risk_level,
            "decision": decision,
            "color": color,
            "risk_factors": risk_factors,
            "shap_explanation": shap_explanation,
            "input_summary": {
                "age": application.age,
                "income": application.income,
                "loan_amount": application.loan_amount,
                "employed_years": application.employed_years,
                "education": application.education,
                "gender": application.gender,
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
