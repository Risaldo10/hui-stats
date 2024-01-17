# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

LOGGER = get_logger(__name__)


def run():
  st.set_page_config(page_title="HUI Stats",page_icon="👋",layout='wide')
  st.write("# Welcome to Stats! 👋")
  if 'hui_training' not in st.session_state:
    hui_training = pd.read_csv('training_data.csv',delimiter=';')
    hui_training['Number of Practices'] = hui_training.iloc[:,3:].sum(axis=1)
    hui_training.set_index('Player', inplace=True)
    hui_training.rename(columns={'Color':'Player Group'},inplace=True)
    hui_training.replace({"Benjamin Alexander Luscombe-Schaumann": "Benjamin Luscombe-Schaumann"}, inplace=True)
    hui_training.replace({"Joakim Hooge Grøndahl Berg": "Joakim Grøndahl Berg"}, inplace=True)
    hui_training.replace({"Achilles Brangstrup Lund-Jacobsen": "Achilles Lund-Jacobsen"}, inplace=True)
    
    st.session_state['hui_training'] = hui_training

  if 'hui_match' not in st.session_state:
    hui_match = pd.read_csv('match_played.csv',delimiter=';')
    hui_match['Number of Matches'] = hui_match.iloc[:,2:].sum(axis=1)
    hui_match['Player'] = [i for i in np.arange(hui_match.shape[0])]
    hui_match.rename(columns={'Color':'Player Group'},inplace=True)

    hui_match.sort_values(['Number of Matches'],ascending=False)
    st.session_state['hui_match'] = hui_match
  
  st.dataframe(st.session_state['hui_training'])
  col1, col2, col3, col4 = st.columns(4)
    
  train_order = ['Name', 'Player Group','Number of Practices','Player']
  custom_palette = {"Blå": "blue", "Hvid": "Gray", "Rød": "red",'Grøn':'Green','Lilla':'Purple','Orange':'Orange'}
  
  

  # Calculate and add mean lines for each group
  with col1:
    num_slicer =st.number_input('Show Last:',min_value=1,max_value=76,value=10 )
    add_mean = st.checkbox('Add averages to graph')
  with col2:
    
    transpose_graph = st.checkbox('Transpose "Axis" in Graph')

  fig, ax = plt.subplots(figsize=(14, 8))
  if transpose_graph: 
    x_name='Name'
    y_name="Number of Practices"
  else:
    x_name="Number of Practices"
    y_name='Name'
  
  all_trainings = st.session_state['hui_training'].iloc[:,2:-1]

  hui_temp = st.session_state['hui_training'][st.session_state['hui_training']['Number of Practices'] > num_slicer]
  st.dataframe(all_trainings.iloc[:,-num_slicer:].sum(axis=1))
  ax = sns.barplot(y=x_name,  x=y_name, data=hui_temp.iloc[num_slicer:].sort_values(['Number of Practices'],ascending=False), hue='Player Group', palette=custom_palette, ax=ax)
  ax.set_title('HUI 2017 - Træning (data fra 14 August 2022 til 18 Januar 2024)')
  ax.tick_params(axis='x', rotation=90) 
  
  if add_mean:
    group_means = hui_temp.groupby('Player Group')['Number of Practices'].mean()
    for i, mean in group_means.items():
      ax.axhline(group_means.loc[i], color=custom_palette[i], linestyle='dashed', linewidth=2)
  st.pyplot(ax.get_figure())
  
    # st.pyplot(ax.get_figure())
  # sns.relplot(x='Player',  y="Total", data=hui, hue='Color') 

if __name__ == "__main__":
    run()
