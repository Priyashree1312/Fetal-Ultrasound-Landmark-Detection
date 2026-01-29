import os

def print_comparison_table():
    # Data from your training sessions
    results = [
        {"Hypothesis": "H1 (Baseline)", "Model": "ResNet18", "Loss": "0.0288", "Status": "Inaccurate"},
        {"Hypothesis": "H2 (Improved)", "Model": "ResNet18", "Loss": "0.0056", "Status": "Accurate"},
        {"Hypothesis": "H3 (Final)",    "Model": "ResNet34", "Loss": "0.0042", "Status": "Best Precision"}
    ]

    print("\n" + "="*65)
    print(f"{'HYPOTHESIS':<20} | {'MODEL':<12} | {'MSE LOSS':<10} | {'RESULT'}")
    print("-" * 65)

    for row in results:
        print(f"{row['Hypothesis']:<20} | {row['Model']:<12} | {row['Loss']:<10} | {row['Status']}")

    print("="*65)
    print("\nCONCLUSION: Hypothesis 3 (ResNet34) is the optimal model.")

if __name__ == "__main__":
    print_comparison_table()