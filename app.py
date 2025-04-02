import pandas as pd
import numpy as np
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

df=pd.read_csv("./heart.csv")
x_train=df.drop(columns='HeartDisease')
y_train=df['HeartDisease']

## Transformation
trf1=ColumnTransformer(
    [
        ('ohe_chest_ecg_st',OneHotEncoder(sparse_output=False,handle_unknown="ignore"),[2,6,10])
    ],remainder='passthrough'
)
trf2=ColumnTransformer(
    [
        ('ohe_sex_exercise',OneHotEncoder(sparse_output=False,handle_unknown="ignore"),[1,8])
    ],remainder='passthrough'
)
trf3=ColumnTransformer([
    ("scale",MinMaxScaler(),slice(0,11))
])
## train the model
from sklearn.tree import DecisionTreeClassifier
trf4 = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=5, min_samples_split=10)


## Creating pipe
pipe=Pipeline(
    [
        ("trf1",trf1),
        ("trf2",trf2),
        ('trf3',trf3),
        ('trf4',trf4)
    ]
)

st.title("Heart Disease Predictor")

##Train the model
pipe.fit(x_train,y_train)

st.sidebar.title("Input Features")
age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()))
sex=['M','F']
Sex=st.selectbox("Sex:",sex)
chestpaintype=['ATA','NAP','TA','ASY']
Chestpaintype=st.selectbox("ChestPainType",chestpaintype)
restingbp= st.sidebar.slider("RestingBP", int(df['RestingBP'].min()), int(df['RestingBP'].max()))
cholesterol= st.sidebar.slider("Cholesterol", int(df['Cholesterol'].min()), int(df['Cholesterol'].max()))
fastingBS=[0,1]
FastingBS=st.selectbox("FastingBS",fastingBS)
restingecg=['Normal','LVH','ST']
Restingecg=st.selectbox("RestingECG",restingecg)
MaxHR= st.sidebar.slider("MaxHR", int(df['MaxHR'].min()), int(df['MaxHR'].max()))
exerciseangina=['N','Y']
ExerciseAngina=st.selectbox("ExerciseAngina",exerciseangina)
Oldpeak= st.sidebar.slider("Oldpeak", float(df['Oldpeak'].min()), float(df['Oldpeak'].max()))
st_slope=['Up','Flat','Down']
ST_Slope=st.selectbox("ST_Slope",st_slope)

input_data=[[age,Sex,Chestpaintype,restingbp,cholesterol,FastingBS,Restingecg,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]]

prediction=pipe.predict(input_data)
if prediction[0]==0:
    prediction_output="Negative"
else:
    prediction_output="Positive"

st.write("Prediction")

st.write(f"Heart Disease Predicted: {prediction_output}")








