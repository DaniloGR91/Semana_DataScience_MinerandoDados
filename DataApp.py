import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Cabeçalho
st.title('Boston House Prices')
st.markdown('A DataApp to show and predict boston house prices')
# Abrindo dataframe
data = pd.read_csv('dataMLfinal.csv', index_col=0)

# DataFrame
st.header('DataFrame')
# Escolha das colunas do dataset
cols = list(data.columns)
selCols = st.multiselect('Parameters for DataFrame: ',
                         cols,
                         default=['RM', 'LSTAT', 'MEDV'])

# CheckBox para mostrar a descrição de cada coluna
paramDescr = open('boston.descr.txt', 'r')
if st.checkbox('Show parameters descriptions'):
    st.text(paramDescr.read())
else:
    paramDescr.close()

# Visualização do DataSet
st.dataframe(data[selCols])

#### Distribution ####
st.header('Distribution')
# Visualização do histograma e boxplot
selHist = st.selectbox(
    'Data for distribution visualization:', cols)

# Função para definir histograma e boxplot


def histBox(df, col):

    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(go.Histogram(
        x=df[col], name='Hitrogram'),
        row=1, col=1)

    fig.add_trace(go.Box(
        y=df[col], name='Boxplot'),
        row=1, col=2)

    fig.update_layout(height=600, width=800,
                      title_text=f'Distribution of {col}')
    return fig


figHist = histBox(data, selHist)
st.write(figHist)

#### ScatterPlot ####
st.header('ScatterPlot')
# Seleção dos eixos do scatterplot
selScat = []
selScat.append(st.selectbox('X axis of ScatterPlot', cols, index=4))
selScat.append(st.selectbox('Y axis of ScatterPlot', cols, index=1))

# Função para definir scatterplot


def scatplot(df, col):
    col = col
    fig = go.Figure(layout={'xaxis': {'title': {'text': col[0]}},
                            'yaxis': {'title': {'text': col[1]}}})
    fig.add_trace(go.Scatter(
        x=df[col[0]],
        y=df[col[1]],
        mode='markers',
        name='ScatterPlot'))

    fig.update_layout(height=600, width=800,
                      title_text=f'ScatterPlot of {col[0]}x{col[1]}')
    return fig


figScat = scatplot(data, selScat)
st.write(figScat)

#### Modelo de Machine Learning ###
# Separando x e y
x = data.drop('MEDV', axis=1)
y = data['MEDV']
# Treinando o modelo
model = RandomForestRegressor()
model.fit(x, y)

#### SideBar para Predictions####
st.sidebar.header('Price Prediction')
predCols = cols.copy()
predCols.remove('MEDV')
st.sidebar.multiselect('Parameteres for prediction',
                       predCols,
                       default=predCols)

paramInput = []
maxParam = {}
for c in cols:
    if c == 'MEDV':
        continue
    maxcol = data[c].max() + data[c].quantile(0.25)
    param = st.sidebar.number_input(c,
                                    min_value=0.0,
                                    max_value=maxcol,
                                    value=data[c].median())
    paramInput.append(param)

btnPredict = st.sidebar.button('Price Predict')

if btnPredict:
    result = model.predict([paramInput])
    st.sidebar.subheader('The predicted value is:')
    result *= 1000
    st.sidebar.markdown(f'US$ {result[0]:.2f}')
