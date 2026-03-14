# ======================================
# IMPORT LIBRARIES
# ======================================

import csv
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Dash imports
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc


# ======================================
# INITIALIZE FLASK APP
# ======================================

app = Flask(__name__)


# ======================================
# LOAD MACHINE LEARNING MODEL
# ======================================

model = joblib.load("churn_pipeline.pkl")


# ======================================
# DASH APP INITIALIZATION
# ======================================

dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dashboard/",
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)


# ======================================
# DASHBOARD LAYOUT
# ======================================

dash_app.layout = dbc.Container([

     html.Div(
        [
            html.A(
                "⬅ Back to Prediction",
                href="/",
                style={
                    "textDecoration": "none",
                    "backgroundColor": "#2c3e50",
                    "color": "white",
                    "padding": "10px 15px",
                    "borderRadius": "6px",
                    "fontWeight": "600"
                }
            )
        ],
        style={"marginBottom": "20px"}
    ),

     html.H1("Customer Churn Analytics Dashboard",
        style={
            "textAlign": "center",
            "marginBottom": "30px",
            "color": "#2c3e50"
        }
             ),

    dcc.Interval(
        id="interval-component",
        interval=5 * 1000,  # refresh every 5 seconds
        n_intervals=0
    ),

    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Total Predictions"),
            html.H2(id="total-predictions")
        ]))),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("High Risk Customers"),
            html.H2(id="high-risk")
        ]))),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Medium Risk Customers"),
            html.H2(id="medium-risk")
        ]))),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Low Risk Customers"),
            html.H2(id="low-risk")
        ])))
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Graph(id="risk-chart")
        )
    ]),

    html.H3("Recent Customer Predictions"),

    dash_table.DataTable(
        id="customer-table",
        columns=[
            {"name": "Contract", "id": "Contract"},
            {"name": "MonthlyCharges", "id": "MonthlyCharges"},
            {"name": "Prediction", "id": "Prediction"},
            {"name": "Probability", "id": "Probability"},
            {"name": "RiskLevel", "id": "RiskLevel"},
        ],
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "center",
            "padding": "10px",
            "fontFamily": "Arial"
        },
        style_header={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontWeight": "bold"
        }
    )

], fluid=True)


# ======================================
# LOAD DASHBOARD DATA
# ======================================

import os

def load_dashboard_data():
    file = "predictions_log.csv"  # make sure this matches the CSV used in your callback

    if not os.path.exists(file):
        return pd.DataFrame()  # return empty DataFrame if file doesn't exist

    try:
        df = pd.read_csv(file)
        # Strip spaces from columns
        df.columns = df.columns.str.strip()

        # It Ensure that  RiskLevel exists even if old CSV is broken
        if "RiskLevel" not in df.columns:
            df["RiskLevel"] = "Unknown"

        return df
    except Exception as e:
        print("Error loading dashboard data:", e)
        return pd.DataFrame()

# Confirm columns

# ======================================
# DASHBOARD CALLBACK (AUTO UPDATE)
# ======================================

@dash_app.callback(
    [
        Output("risk-chart", "figure"),
        Output("customer-table", "data"),
        Output("total-predictions", "children"),
        Output("high-risk", "children"),
        Output("medium-risk", "children"),
        Output("low-risk", "children"),
    ],
    Input("interval-component", "n_intervals")
)

def update_dashboard(n):

    df = load_dashboard_data()

    if df.empty:
        # Return empty chart and zero metrics if no data
        empty_chart = px.bar(title="No Data Available")
        return empty_chart, [], "0", "0", "0", "0"

    # KPI Metrics
    total_predictions = len(df)
    high_risk = len(df[df["RiskLevel"] == "High Risk"])
    medium_risk = len(df[df["RiskLevel"] == "Medium Risk"])
    low_risk = len(df[df["RiskLevel"] == "Low Risk"])

    # Risk chart
    risk_df = df["RiskLevel"].value_counts().reset_index()
    risk_df.columns = ["RiskLevel", "Customers"]

    risk_chart = px.bar(
        risk_df,
        x="RiskLevel",
        y="Customers",
        title="Customer Risk Segmentation"
    )

    # Table
    table_data = df[[
        "Contract",
        "MonthlyCharges",
        "Prediction",
        "Probability",
        "RiskLevel"
    ]].sort_values(by="Probability", ascending=False)

    return (
        risk_chart,
        table_data.to_dict("records"),
        total_predictions,
        high_risk,
        medium_risk,
        low_risk
    )


# ======================================
# FLASK ROUTES
# ======================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Collect form data
    data = {
        "gender": request.form.get("gender"),
        "SeniorCitizen": request.form.get("SeniorCitizen"),
        "Partner": request.form.get("Partner"),
        "Dependents": request.form.get("Dependents"),
        "tenure": int(request.form.get("tenure", 0)),
        "PhoneService": request.form.get("PhoneService"),
        "MultipleLines": request.form.get("MultipleLines"),
        "InternetService": request.form.get("InternetService"),
        "OnlineSecurity": request.form.get("OnlineSecurity"),
        "OnlineBackup": request.form.get("OnlineBackup"),
        "DeviceProtection": request.form.get("DeviceProtection"),
        "TechSupport": request.form.get("TechSupport"),
        "StreamingTV": request.form.get("StreamingTV"),
        "StreamingMovies": request.form.get("StreamingMovies"),
        "Contract": request.form.get("Contract"),
        "PaperlessBilling": request.form.get("PaperlessBilling"),
        "PaymentMethod": request.form.get("PaymentMethod"),
        "MonthlyCharges": float(request.form.get("MonthlyCharges", 0)),
        "TotalCharges": float(request.form.get("TotalCharges", 0))
    }

    df = pd.DataFrame([data])

    # Model prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    prob_percent = round(probability * 100, 2)

    # Risk segmentation
    if prob_percent < 40:
        risk = "Low Risk"
    elif prob_percent < 70:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    label = "Churn" if prob_percent >= 50 else "Stay"

    # Save prediction
    log_data = [
        data["gender"],
        data["SeniorCitizen"],
        data["Partner"],
        data["Dependents"],
        data["tenure"],
        data["PhoneService"],
        data["InternetService"],
        data["Contract"],
        data["MonthlyCharges"],
        data["TotalCharges"],
        label,
        prob_percent,
        risk
    ]

    header = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "InternetService",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "Prediction",
        "Probability",
        "RiskLevel"
    ]

    file = "predictions_log.csv"
    file_exists = os.path.isfile(file)

    with open(file, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(log_data)

    return render_template(
        "result.html",
        prediction=label,
        probability=prob_percent
    )


# ======================================
# RUN SERVER
# ======================================

if __name__ == "__main__":
   port = int(os.environ.get("PORT", 5000))
   app.run(host="0.0.0.0", port=port)