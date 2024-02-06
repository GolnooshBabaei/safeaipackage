import pandas as pd

def read_employeedata():
    df = pd.read_excel("datasets/employee.xlsx")
    print(df.head())
    return df