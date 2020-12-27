# Assignment 4 - Aravind
# visit- https://assig4-icfoss.herokuapp.com/

# import libraries
import matplotlib.pyplot as plt 
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns 

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("""

### Machine Learning With FOSS

## Assignment 4
### Aravind
#### Analysis of datasets related to red and white variants of the Portuguese "Vinho Verde" wine.
Two datasets were created, using red and white wine samples.
  - The inputs include objective tests (e.g. PH values) and the output is based on sensory data(median of at least 3 evaluations made by wine experts). Each expert graded the wine quality between 0 (very bad) and 10 (very excellent).

Attribute information:

Input variables (based on physicochemical tests):
 > fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

 Output variable (based on sensory data): 
> quality (score between 0 and 10)
""")


st.markdown("## Analysis of red wine data")

red_wine = pd.read_csv('winequality-red.csv', sep=';')

st.markdown("""### Sample of the data
- dataset has 1599 unique values
""")

st.table(red_wine.head())

st.markdown("""### Features of the data like
- mean values of attributes
- maximum and minimum values of attributes
""")

# red_wine.info()
if st.checkbox('click to view the data'):
  st.table(red_wine.describe())

st.markdown("""- maximum quality of red wine here is 8 on a scale of 10.""")

# red_wine.columns

st.markdown("""### Properties of red wine samples with maximum quality""")

quality_red = red_wine.where(red_wine['quality'] == 8).dropna()
st.table(quality_red.reset_index())

st.markdown("""### pH value variation of white wine samples
1 for high quality white wine

2 for the whole white wine sample set
""")
st.subheader("pH variation of high quality wine")
quality_red['pH'].plot()
st.pyplot()

st.subheader("pH variation of red wine samples")
plt.figure(figsize=(12,6))
red_wine[ 'pH'].plot()
st.pyplot()

st.markdown("""### Distribution of **pH** of the wine samples""")

sns.displot(red_wine['pH'])
st.pyplot()

"""### Relation between alcohol content and wine quality
 - preceived wine quality, by tasters, is higher for wine with higher alcohol content
"""

sns.jointplot(x='quality', y='alcohol', data=red_wine, kind='hist')
st.pyplot()

# """### Correlation matrix for the attributes of red wine"""

# red_wine.corr()

"""### Heat map for the correlation matrix
- perceived quality has considerable positive relation with alcohol content, citric acid content, and sulphates.
"""
if st.checkbox('click to view the heat map'):
  plt.figure(figsize=(12,12))
  sns.heatmap(red_wine.corr(), cmap='coolwarm', annot=True)
  st.pyplot()

"""## Analysis of white wine data"""

white_wine = pd.read_csv('winequality-white.csv', sep=';')

"""### Sample of the data"""

st.table(white_wine.head(6))

"""### Features of the data like
- mean values of attributes
- maximum and minimum values of attributes
"""

# white_wine.info()

"""- white wine dataset has larger number of 4898 samples, so the analysis can be more accurate"""
if st.checkbox('click here to view the data'):
  st.table(white_wine.describe())

"""### Properties of white wine samples with maximum quality"""

quality_white = white_wine.where(white_wine['quality'] == 9).dropna()
st.table(quality_white.reset_index())

"""### __pH__ value variation of white wine samples
- for high quality white wine
- for the whole white wine sample set
"""

st.subheader("pH variation of high quality white wine")
quality_white['pH'].plot()
st.pyplot()

st.subheader("pH variation of hite wine samples")
plt.figure(figsize=(15,6))
white_wine[ 'pH'].plot()
st.pyplot()

"""### Distribution of pH of the wine samples"""

sns.displot(white_wine['pH'])
st.pyplot()

"""### Relation between alcohol content and wine quality
- preceived wine quality, by tasters, is higher for wine with higher alcohol content
"""

sns.jointplot(x='quality', y='alcohol', data=white_wine, kind='hist')
st.pyplot()

"""### Correlation matrix for the attributes of white wine

---


"""

# white_wine.corr()

"""### Heat map for the correlation matrix
- perceived quality has considerable positive relation with alcohol content, and inverse relation with density of the wine
"""
if st.checkbox('click the box to view the heatmap'):
  plt.figure(figsize=(12,12))
  sns.heatmap(white_wine.corr(), cmap='coolwarm', annot=True)
  st.pyplot()

"""## Use case
- above dataset can be used to predict physiochemical features of wine to get a higher quality product.

## Dataset source

>  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

---
"""

