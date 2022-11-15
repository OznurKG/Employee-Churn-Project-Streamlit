import pickle
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Churn App")

#Import the dataset
employee = pd.read_csv("HR_Dataset.csv")
df = employee.copy()

#Clean columns
df = df.iloc[:, [0,1,2,3,4,5,7,8,9,6]]
df = df.rename(columns={"Departments " : "departments", "Work_accident" : "work_accident"})
model = pickle.load(open("final_churn_model", "rb"))

#separating X and y
X = df.drop("left", axis=1)
y = df["left"]

#Adding background image from your-local
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('employee.jpg') 

# Adding background image from url
# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://img.freepik.com/free-vector/business-doodle-vector-human-resources-concept_53876-126582.jpg?w=996&t=st=1667582934~exp=1667583534~hmac=d65d1b36f1eb5cb85128167529e735d1e39c9be971a0cc556aa4ce4f339e9df5");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# add_bg_from_url() 

st.markdown("<h1 style='text-align: center;border: solid; color: black;'>Employee Churn Prediction App</h1>", unsafe_allow_html=True)
st.write("""
This app is created to predict **Employee Churn**. Here employee churn means the employee leaves the job.

     """)


#To download the dataset    
def filedownload(df):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions

    href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'

    return href

st.markdown(filedownload(df), unsafe_allow_html=True)

#Subheader
st.subheader('User Input Features')


#Sidebar background image from local
def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
side_bg = 'bluee.png'
sidebar_bg(side_bg)

#Sidebar header
st.sidebar.header('User Input Features')

#To show uploaded data or original dataset
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
   st.table(df.head())
else:
   st.markdown("<p5 style='text-align: left;color: black; background-color: #b5e7a0'>Awaiting CSV file to be uploaded or filters on the sidebar to be selected. Currently using example input parameters (Please tick the checkbox to see).</p>", unsafe_allow_html=True)

   #To show data   
   cbox = st.checkbox("Show Data")

   if cbox:
      st.table(df.sample(5))

#Create features on the sidebar
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():

        
        satisfaction_level = st.sidebar.slider('Satisfaction Level', float(df["satisfaction_level"].min()), float(df["satisfaction_level"].max()), float(0.48), 0.01)
        last_evaluation = st.sidebar.slider('Last Evaluation', float(df["last_evaluation"].min()), float(df["last_evaluation"].max()), float(0.63), 0.01)
        average_montly_hours = st.sidebar.number_input('Average Monthly Hours', int(df["average_montly_hours"].min()),int(df["average_montly_hours"].max()), int(226), 1)
        number_project = st.sidebar.number_input('Number of Project', int(df["number_project"].min()), int(df["number_project"].max()), int(7), 1)
        time_spend_company = st.sidebar.number_input('Time Spent in the Company', int(df["time_spend_company"].min()), int(df["time_spend_company"].max()), int(3), 1)
        work_accident = st.sidebar.radio('Work Accident', ('Yes', 'No'))
        if work_accident == 'Yes':
            work_accident = 1
        else:
            work_accident = 0
        promotion_last_5years = st.sidebar.radio('Promotion in the Last 5 Years', ('Yes', 'No'))
        if promotion_last_5years == 'Yes':
            promotion_last_5years = 1
        else:
            promotion_last_5years = 0
        sorted_departments = ["IT", "Research and Development","Accounting", "Human Resources", "Management", "Marketing", "Product Management","Sales", "Support", "Technical"]
        departments = st.sidebar.selectbox('Department', sorted_departments)
        if departments == "IT":
            departments = "IT"
        elif departments == "Research and Development":
            departments = "RandD"
        elif departments == "Accounting":
            departments = "accounting"
        elif departments == "Human and Resources":
            departments = "hr"
        elif departments == "Management":
            departments = "management"
        elif departments == "Product Management":
            departments = "product_mng"
        elif departments == "Sales":
            departments = "sales"
        elif departments == "Support":
            departments = "sales"
        elif departments == "Technical":
            departments = "technical"
        salary_level = df["salary"].str.title().unique()
        salary = st.sidebar.selectbox('Salary', salary_level)
        if salary == "Low":
            salary ="low"
        elif salary =="Medium":
            salary="medium"
        else:
            salary="high"
        data= {'satisfaction_level' : satisfaction_level,
                'last_evaluation' : last_evaluation,
                'number_project' : number_project,
                'average_montly_hours' : average_montly_hours,
                'time_spend_company' : time_spend_company,
                'work_accident' : work_accident,
                'promotion_last_5years' : promotion_last_5years,
                'departments' : departments,
                'salary' : salary
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

#To see the selected filters on the main page
col1, col2, col3, col4 = st.columns(4)
col1.metric("*Satisfaction Level", input_df["satisfaction_level"]) 
col2.metric("*Last Evaluation", input_df["last_evaluation"])
col3.metric("*Number Project", input_df["number_project"])
col4.metric("*Average Montly Hours", input_df["average_montly_hours"])
col1, col2, col3,col4 = st.columns(4)
col1.metric("*Time Spent Company", input_df["time_spend_company"]) 
col2.metric("*Work Accident", input_df["work_accident"])
col3.metric("*Promotion in Last 5 Years", input_df["promotion_last_5years"])
col4.metric("*Salary", input_df["salary"][0].title())
col1, col2 =st.columns(2)
with col1:
    if input_df["departments"][0] in ["IT", "RandD", "product_mng", "hr"]:
        if input_df["departments"][0] == "IT":                                     
            col1.metric("*Department", "IT")
        elif input_df["departments"][0] == "RandD":
            col1.metric("*Department", "Research and Development")
        elif input_df["departments"][0] == "product_mng":
             col1.metric("*Department", "Product Management")
        else:
             col1.metric("*Department", "Human Resources")
    else:
        col1.metric("*Department", input_df["departments"][0].title())
st.markdown("---")


#Check button and results on the sidebar
st.sidebar.write("Press **check** if configuration is complete.")
sample = input_df
if st.sidebar.button("Check"):
    prediction = model.predict(sample)
    prediction_proba = model.predict_proba(sample)
    if prediction == 0 :
        st.subheader("Prediction")
        result = f'<p style="color:black; border-color:#8dc6ff; font-size: 24px; background-color:#b5e7a0">The prediction for the employee is <b>Ongoing</b> with the {prediction_proba[:,0][0]*100 : .1f}% probability.</p>'
        st.markdown(result, unsafe_allow_html=True)
                        
    else:
        st.subheader("Prediction")
        result = f'<p style="color:black; border-color:#8dc6ff; font-size: 24px; background-color:#f7786b">The prediction for the employee is <b>Left</b> with the {prediction_proba[:,1][0]*100 : .1f}% probability.</p>'
        st.markdown(result, unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = (prediction_proba[:,1][0])*100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk (%)",'font': {'size': 24}},
            gauge = {'axis': {'range': [None, 100]},
                    'bar' : {'color':'red'}, 
                    'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}
                    }))      
    st.plotly_chart(fig, use_container_width=True)

st.markdown('**Created by G-7 using Streamlit**')
