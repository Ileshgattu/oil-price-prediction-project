#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Oil_Price_Prediction_App.ipynb

'''

import numpy as np
import streamlit as st
from keras.models import model_from_json
import datetime
import json
import os

# Load json and create model
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('model.h5')
print('loaded model from disk')

# Last 3days prices required to make input for future prediction

def price_prediction(days):
    x_input = np.array([23.41,23.68,22.89])
    temp_input = list(x_input)
    lst_output = []
    
    i = 0
    while(i<days):
        
        if len(temp_input)>3:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape([1,3,1])
            
            yhat = loaded_model.predict(x_input,verbose=0)
            #print('Day'{} Price Prediction {}'.format(i, yhat)')
            
            temp_input.append(yhat[0][0])
            temp_input = temp_input[1:]
            
            lst_output.append(yhat[0][0])
            i = i+1
            
        else:
            x_input = x_input.reshape((1,3,1))
            yhat = loaded_model.predict(x_input,verbose=0)
            # print('Date {} Price Prediction {}'.format(i, yhat))
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])
            i = i+1
    return lst_output

def main():
    
    # Giving a title
    st.title('Oil Price Prediction : LSTM Model')
    
    # To display date in UI
    today = datetime.datetime.now()
    next_day = today.day+1
    min_date = datetime.date(today.year, today.month, next_day)
    max_date = datetime.date(today.year, today.month, next_day+8)
    
    start_date = st.date_input('Start date',min_value=min_date,max_value=max_date, value=min_date)
    end_date = st.date_input('End date', min_value=min_date, max_value = max_date, value=max_date)
    
    if start_date <= end_date:
        st.success('Start date: {}\n\nEnd date: '.format(start_date, end_date))
    else:
        st.error('Error: End date must be after start date.')
    total_days = end_date.day-today.day
    no_days = end_date.day-start_date.day+1
    submit = st.button('Predict Price')
    if submit:
        
        st.subheader('Predicted Price for Crude Oil')
        
        # Code for Prediction
        lst_output = []
        lst_output = price_prediction(total_days)
        # creating a button for prediction
        for i in range(total_days-no_days+1,total_days+1):
            st.write('Date', datetime.date(today.year, today.month, today.day+i) , 'Price Prediction:',lt_output[i-1])
            print(i)
            
if __name__ == '__main__':
    main()


# In[ ]:




