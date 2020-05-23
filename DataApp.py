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
data = pd.read_csv('dataML.csv')

# Escolha das colunas do dataset
cols = []
for c in data.columns:
    cols.append(c)
selCols = st.multiselect('Parameters for dataset: ',
                         cols,
                         default=['RM', 'LSTAT', 'MEDV'])

# CheckBox para mostrar a descrição de cada coluna
if st.checkbox('Show parameters descriptions'):
    st.text(boston.DESCR[boston.DESCR.find(
        ':Attribute'):boston.DESCR.find(':Missing')])

# Visualização do DataSet
st.dataframe(data[selCols])

# Visualização do histograma
selHist = st.selectbox(
    'Data for histogram visualization:', cols)
fig = px.histogram(data, x=data[selHist])
st.write(fig)
