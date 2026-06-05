import pandas as pd
import joblib
import streamlit as st
model=joblib.load("credit_deafult_model.joblib")
st.title("Credit Default Risk Prediction")
st.write("Enter customer details")
RevolvingUtilizationOfUnsecuredLines =st.number_input("Revolving Utilization Of Unsecured Lines ",min_value=0.0)
age=st.number_input("Age ",min_value=18)
NumberOfTime30_59DaysPastDueNotWorse=st.number_input("Number Of Time 30-59 Days Past Due Not Worse",min_value=0)
DebtRatio=st.number_input("Debt Ration",min_value=0.0)
MonthlyIncome=st.number_input("Monthly Income",min_value=0.0)
NumberOfOpenCreditLinesAndLoans=st.number_input("Number Of Open Credit Lines And Loans",min_value=0.0)
NumberOfTimes90DaysLate=st.number_input("Number Of Times 90 Days Late",min_value=0)
NumberRealEstateLoansOrLines=st.number_input("Number Real Estate Loans Or Lines",min_value=0.0)
NumberOfTime60_89DaysPastDueNotWorse=st.number_input("Number Of Time 60-89 Days Past Due Not Worse",min_value=0.0)
NumberOfDependents=st.number_input("Number Of Dependents",min_value=0.0)
if st.button("Predict"):
    input_df=pd.DataFrame({"RevolvingUtilizationOfUnsecuredLines":[RevolvingUtilizationOfUnsecuredLines],
                          "age":[age],
                          "NumberOfTime30-59DaysPastDueNotWorse":[NumberOfTime30_59DaysPastDueNotWorse],
                          "DebtRatio":[DebtRatio],
                          "MonthlyIncome":[MonthlyIncome],
                          "NumberOfOpenCreditLinesAndLoans":[NumberOfOpenCreditLinesAndLoans],
                          "NumberOfTimes90DaysLate":[NumberOfTimes90DaysLate],
                          "NumberRealEstateLoansOrLines":[NumberRealEstateLoansOrLines],
                           "NumberOfTime60-89DaysPastDueNotWorse":[NumberOfTime60_89DaysPastDueNotWorse],
                           "NumberOfDependents":[NumberOfDependents]
                           
                          })
    prediction=model.predict(input_df)[0]
    probability=model.predict_proba(input_df)[0][1]
    if prediction==1:
        st.error("High Risk Of Default")
    else:
        st.success("Low Risk of Default")
    st.write(f"Default Probability:{probability: .2%}")







