import pandas as pd

def create_seating_excel():
    seats = [f"Seat-{i}" for i in range(1, 51)]

    df = pd.DataFrame({
        "Seat_No": seats,
        "Assigned_Roll_No": "",
        "Verification_Status": "Pending"
    })

    df.to_excel("seating_plan.xlsx", index=False)
    print("Seating plan created!")

if __name__ == "__main__":
    create_seating_excel()
