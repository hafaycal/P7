
import streamlit as st

# Utilisation de SK_IDS dans st.sidebar.selectbox
import seaborn as sns
import os
import plotly.express as px
import streamlit as st
from PIL import Image
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import time
import pickle
import plotly.graph_objects as go
import shap
from streamlit_shap import st_shap
import numpy as np
###---------- load data -------- 


def test_load():
    assert load_dataset(1).size == 5000  
