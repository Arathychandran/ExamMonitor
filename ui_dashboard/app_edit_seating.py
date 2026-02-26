# ui_dashboard/app_edit_seating.py
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "dev_secret"  # change for production

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "..", "seating_plan.xlsx")

def load_df():
    return pd.read_excel(EXCEL_PATH, dtype=str).fillna("")

def save_df(df):
    df.to_excel(EXCEL_PATH, index=False)

@app.route("/", methods=["GET"])
def index():
    df = load_df()
    rows = df.to_dict(orient="records")
    return render_template("edit_seating.html", rows=rows, columns=list(df.columns))

@app.route("/save", methods=["POST"])
def save():
    # Build dataframe from form data
    seat_nos = request.form.getlist("Seat_No")
    assigned = request.form.getlist("Assigned_Roll_No")
    status_list = request.form.getlist("Verification_Status")
    rows = []
    for s, a, st in zip(seat_nos, assigned, status_list):
        rows.append({
            "Seat_No": s,
            "Assigned_Roll_No": a,
            "Verification_Status": st
        })
    df = pd.DataFrame(rows)
    save_df(df)
    flash("Seating plan saved successfully.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
