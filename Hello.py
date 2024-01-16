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

LOGGER = get_logger(__name__)


def run():
  st.set_page_config(page_title="HUI Stats",page_icon="👋",layout='wide')

  st.write("# Welcome to Stats! 👋")
  if 'hui_training' not in st.session_state:
    hui_training = pd.read_csv('trainings_data.csv',delimiter=';')
    hui_training['Number of Practices'] = hui_training.iloc[:,2:].sum(axis=1)
    hui_training['Player'] = [i for i in np.arange(hui_training.shape[0])]
    hui_training.index.name = 'Player'  
    hui_training['Player Name'] = hui_training['Name'].str.split()
    hui_training['Player Name'] = hui_training['Player Name'].apply(lambda x: ' '.join([x[i] for i in range(len(x)) if i != 1 or len(x) < 4]))
    hui_training  = hui_training.drop('Name',axis=1)
    hui_training.rename(columns={'Color':'Player Group','Player Name':'Name'},inplace=True)
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
  with col1:
     st.date_input('Start Date:', )
  
  train_order = ['Name', 'Player Group','Number of Practices','Player']
  custom_palette = {"Blue": "blue", "White": "green", "Red": "red"}
  
  hui_temp = st.session_state['hui_training'][st.session_state['hui_training']['Number of Practices'] > 15]

  fig, ax = plt.subplots(figsize=(14, 8)) 
  ax = sns.barplot(x='Name',  y="Number of Practices", data=hui_temp.sort_values(['Number of Practices'],ascending=False), hue='Player Group', palette=custom_palette, ax=ax)
  ax.set_title('HUI 2017 - Træning (data fra 1 Januar 2022 til 13 November 2023)')
  ax.tick_params(axis='x', rotation=70) 
  
  # Calculate and add mean lines for each group
  add_mean = st.checkbox('Add averages to graph')
  if add_mean:
    group_means = hui_temp.groupby('Player Group')['Number of Practices'].mean()
    for i, mean in group_means.items():
      ax.axhline(group_means.loc[i], color=custom_palette[i], linestyle='dashed', linewidth=2)
  st.pyplot(ax.get_figure())
  
    # st.pyplot(ax.get_figure())
  # sns.relplot(x='Player',  y="Total", data=hui, hue='Color') 

if __name__ == "__main__":
    run()
