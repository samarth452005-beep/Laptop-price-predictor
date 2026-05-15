import  streamlit as st
import pickle
import numpy as np


st.title("laptop-price-predictor")
import sklearn

#import the model
import joblib

pipe = joblib.load('model.pkl')
df=pickle.load(open('df.pkl','rb'))

#brands
company = st.selectbox('Brand',df['Company'].unique())

#type of laptop
Type = st.selectbox('Type',df['TypeName'].unique())

#ram
Ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

#weight
weight = st.number_input('weight of the laptop')

#touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

#Ips
ips = st.selectbox('IPS',['No','Yes'])

#screensize
screen_size = st.number_input('screen size')

#resolution
resolution = st.selectbox('Resolution',[    "1024x768",
    "1280x720",
    "1280x800",
    "1366x768",
    "1440x900",
    "1600x900",
    "1920x1080",
    "1920x1200",
    "2160x1440",
    "2256x1504",
    "2304x1440",
    "2400x1600",
    "2560x1440",
    "2560x1600",
    "2880x1800",
    "3000x2000",
    "3072x1920",
    "3200x1800",
    "3240x2160",
    "3456x2160",
    "3840x2160"])

#cpu
cpu = st.selectbox('CPU',df['Cpu_brand'].unique())

#hdd
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

#SSd
ssd = st.selectbox('SSD(in GB)',[0,128,256,512,1024,2048])

#gpu
gpu = st.selectbox('GPU',df['Gpu_brands'].unique())

#os
os = st.selectbox('OS',df['Os'].unique())

if st.button('Predict_price'):

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])

    ppi = ((x_res**2) + (y_res**2))**0.5 / screen_size

    import pandas as pd

    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [Type],
        'Ram': [Ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'Ips': [ips],
        'ppi': [ppi],
        'Cpu_brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu_brands': [gpu],
        'Os': [os]
    })

    prediction = pipe.predict(query)

    st.title(f"Predicted Price: ₹{int(np.exp(prediction[0]))}")
