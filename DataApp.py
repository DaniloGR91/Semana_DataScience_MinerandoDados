import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor

# Cabeçalho
st.title('Boston House Prices')
st.markdown('A DataApp to show and predict boston house prices')
# Abrindo dataframe
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target

# Um modo que possa escolher quais colunas estarão no df
cols = ['RM', 'AGE', 'LSTAT', 'MEDV']
# Colocar uma caixa escondida que só aparece quando a pessoa clicar (popup?) com boston.descr
st.dataframe(data[cols])

# gráfico teste
fig = px.histogram(data, x=data['MEDV'])
st.write(fig)
