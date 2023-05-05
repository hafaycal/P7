# TO RUN : $streamlit run dashboard/dashboard.py
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.50:8501
# Online URL : http://15.188.179.79

#

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
def load_all_data(sample_size):
    
    data = pd.read_csv("data/df_final.csv",nrows=sample_size)
    #X_train = pd.read_csv("data/X_train_transformed.csv",nrows=sample_size)
    X_test = pd.read_csv("data/X_test_transformed.csv",nrows=sample_size)
    #train_set = pd.read_csv('data/application_train.csv',nrows=sample_size)
    #train_set = pd.read_csv("data/X_train_transformed.csv",nrows=sample_size)
    X_train = pd.read_csv("data/X_test_clean.csv",nrows=sample_size)
    y_pred_test_export = pd.read_csv("data/y_pred_test_export.csv")

    #Preparation des données age
    data['DAYS_BIRTH']= data['DAYS_BIRTH']/-365
    bins= [0,10,20,30,40,50,60,70,80]
    data['age_bins'] = (pd.cut(data['DAYS_BIRTH'], bins=bins)).astype(str)

    
 
        
    return data ,y_pred_test_export,train_set

#------------- Affichage des infos client en HTML------------------------------------------
def display_client_info(id,revenu,age,nb_ann_travail):
   
    components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div class="card" style="width: 500px; margin:10px;padding:0">
        <div class="card-body">
            <h5 class="card-title">Info Client</h5>
            
            <ul class="list-group list-group-flush">
                <li class="list-group-item"> <b>ID                           : </b>"""+id+"""</li>
                <li class="list-group-item"> <b>Revenu                       : </b>"""+revenu+"""</li>
                <li class="list-group-item"> <b>Age                          : </b>"""+age+"""</li>
                <li class="list-group-item"> <b>Nombre d'années travaillées  : </b>"""+nb_ann_travail+"""</li>
            </ul>
        </div>
    </div>
    """,
    height=300
    
    )
#==============================================================================================
def predict():
    lgbm = pickle.load(open("lgbm.pkl", 'rb'))
    if lgbm:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)           
            #X_transformed = preprocessing(query)
            y_pred = randomForest.predict(X_train)
            y_proba = randomForest.predict_proba(X_train)
            
            return jsonify({'prediction': y_pred,'prediction_proba':y_proba[0][0]})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')

def predictByClientId():
    lgbm = pickle.load(open("lgbm.pkl", 'rb'))
    if lgbm:
        try:
            json_ = request.json
            print(json_)
            sample_size = 10000
            
            print(json_)  

            sample_size= 20000
            #data_set = data = pd.read_csv("df_final.csv",nrows=sample_size)
            client=data_set[X_train['SK_ID_CURR']==json_['SK_ID_CURR']].drop(['SK_ID_CURR','TARGET'],axis=1)
            print(client)
 
            
            preproc = pickle.load(open("preprocessor.sav", 'rb'))
            #X_transformed =preproc.transform(client)
            y_pred = randomForest.predict(client)
            y_proba = randomForest.predict_proba(client)
            
            return jsonify({'prediction': str(y_pred[0]),'prediction_proba':str(y_proba[0][0])})


        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')


def load_dataset(sample_size):
    df_final = pd.read_csv("data/df_final.csv",nrows=sample_size)
    return df_final

'''Load fitted preprocessor '''
def load_preprocessor():
    preproc = pickle.load(open("models/preprocessor.sav", 'rb'))
    return preproc

'''Transformer X avec le preprocessor '''
def preprocessing(X):
    preprocessor  = load_preprocessor()
    return preprocessor.transform(X)

'''Charger un modele entrainné '''
def load_model(model_to_load):
    if model_to_load == "randomForest":
        model = pickle.load(open("models/classifier_rf_model.sav", 'rb'))
    elif model_to_load == "lgbm":
        model = pickle.load(open("models/model_LGBM.pkl", 'rb'))
    else:
        print("modèle non connu ! Merci de chois : lgbm ou randomForest")
    
    return model

'''Prédire un client avec un modele  '''
def predict_client(model,X):
    X = X.drop(['SK_ID_CURR'],axis=1)
    #X_transformed = preprocessing(X)
    model = load_model(model)
    
    #y_pred = model.predict(X_transformed)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    #y_proba = model.predict_proba(X)

    return y_pred,y_proba


'''Prédire un client par son ID dans le dataset '''
def predict_client_par_ID(model_to_use,id_client):
    sample_size= 20000
    data_set = load_dataset(sample_size)
    client=data_set[data_set['SK_ID_CURR']==id_client].drop(['SK_ID_CURR','TARGET'],axis=1)
    print(client)
    #client_preproceced = preprocessing(client)
    model = load_model(model_to_use)

    y_pred = model.predict(client)
    #y_pred = model.predict(client)
    
    y_proba = model.predict_proba(client)
    #y_proba = model.predict_proba(client)
    return y_pred,y_proba

def get_sk_id_list():
        API_URL = "http://127.0.0.1:5000/api/"

        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"

        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)

        # Convert from JSON format to Python dict
        content = json.loads(response.content)

        # Getting the values of SK_IDS from the content
        SK_IDS = content['data']

        return SK_IDS
    
        

    ##################################################
    ##################################################
    ##################################################

### Data
def show_data(data):
    st.write(data.head(10))

    print("je suis dans la fonction")
### Solvency
def pie_chart(thres,data):
    #st.write(100* (data['TARGET']>thres).sum()/data.shape[0])
    print("je suis dans la fonction")
    percent_sup_seuil =100* (data['TARGET']>thres).sum()/data.shape[0]
    percent_inf_seuil = 100-percent_sup_seuil
    d = {'col1': [percent_sup_seuil,percent_inf_seuil], 'col2': ['% Non Solvable','% Solvable',]}
    df = pd.DataFrame(data=d)
    #fig = plt.pie(df,values='col1', names='col2', title=' Pourcentage de solvabilité des clients di dataset')
    fig = plt.pie(df)
    #plt.pie(y, labels = mylabels, colors=colors,explode = explodevalues, autopct='%1.1f%%', shadow = True)

    st.plotly_chart(fig)

def show_overview(data):
    st.title("Risque")
    risque_threshold = st.slider(label = 'Seuil de risque', min_value = 0.0,
                    max_value = 1.0 ,
                     value = 0.5,
                     step = 0.1)
    #st.write(risque_threshold)
    pie_chart(risque_threshold,data) 

 
### Graphs
def filter_graphs():
    st.subheader("Filtre des Graphes")
    col1, col2,col3 = st.columns(3)
    is_educ_selected = col1.radio("Graph Education",('non','oui'))
    is_statut_selected = col2.radio('Graph Statut',('non','oui'))
    is_income_selected = col3.radio('Graph Revenu',('non','oui'))

    return is_educ_selected,is_statut_selected,is_income_selected

def hist_graph ():
    st.bar_chart(data['DAYS_BIRTH'])
    df = pd.DataFrame(data[:200],columns = ['DAYS_BIRTH','AMT_CREDIT'])
    df.hist()
    st.pyplot()

def education_type(train_set):
    ed = train_set.groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
    u_ed = train_set.NAME_EDUCATION_TYPE.unique() 
    fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
    st.plotly_chart(fig)
##
##
##    fig = plt.Figure(data=[sns.barplot(
##            x=u_ed,
##            y=ed
##        )])
##    fig.update_layout(title_text='Data education')
##
##    st.plotly_chart(fig)
##
##    ed_solvable = train_set[train_set['TARGET']==0].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
##    ed_non_solvable = train_set[train_set['TARGET']==1].groupby('NAME_EDUCATION_TYPE').NAME_EDUCATION_TYPE.count()
##    u_ed = train_set.NAME_EDUCATION_TYPE.unique() 
##
##    #fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
##    #st.plotly_chart(fig)
##
##    fig = plt.Figure(data=[
##        sns.barplot(name='Solvable',x=u_ed,y=ed_solvable),
##        sns.barplot(name='Non Solvable',x=u_ed,y=ed_non_solvable) 
##        ])
##    fig.update_layout(title_text='Solvabilité Vs education')
##
##    st.plotly_chart(fig)

##def statut_plot (train_set):
##    ed = train_set.groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
##    u_ed = train_set.NAME_FAMILY_STATUS.unique() 
##    #fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
##    #st.plotly_chart(fig)
##
##    fig = go.Figure(data=[go.Bar(
##            x=u_ed,
##            y=ed
##        )])
##    fig.update_layout(title_text='Data situation familiale')
##
##    st.plotly_chart(fig)
##
##    ed_solvable = train_set[train_set['TARGET']==0].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
##    ed_non_solvable = train_set[train_set['TARGET']==1].groupby('NAME_FAMILY_STATUS').NAME_FAMILY_STATUS.count()
##    u_ed = train_set.NAME_FAMILY_STATUS.unique() 
##    #fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
##    #st.plotly_chart(fig)
##
##    fig = go.Figure(data=[
##        go.Bar(name='Solvable',x=u_ed,y=ed_solvable),
##        go.Bar(name='Non Solvable',x=u_ed,y=ed_non_solvable) 
##        ])
##    fig.update_layout(title_text='Solvabilité Vs situation familiale')
##
##    st.plotly_chart(fig)



##def income_type (train_set):
##    ed = train_set.groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
##    u_ed = train_set.NAME_INCOME_TYPE.unique() 
##    #fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
##    #st.plotly_chart(fig)
##
##    fig = go.Figure(data=[go.Bar(
##            x=u_ed,
##            y=ed
##        )])
##    fig.update_layout(title_text='Data Type de Revenu')
##
##    st.plotly_chart(fig)
##
##    ed_solvable = train_set[train_set['TARGET']==0].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
##    ed_non_solvable = train_set[train_set['TARGET']==1].groupby('NAME_INCOME_TYPE').NAME_INCOME_TYPE.count()
##    u_ed = train_set.NAME_INCOME_TYPE.unique() 
##    #fig = plt.bar(u_ed, ed, bottom=None, color='blue', label='Education')
##    #st.plotly_chart(fig)
##
##    fig = go.Figure(data=[
##        go.Bar(name='Solvable',x=u_ed,y=ed_solvable),
##        go.Bar(name='Non Solvable',x=u_ed,y=ed_non_solvable) 
##        ])
##    fig.update_layout(title_text='Solvabilité Vs Type de Revenu')
##
##    st.plotly_chart(fig)
##
##
##
#####------------------------ Distribution ------------------------
##def filter_distribution():
##    st.subheader("Filtre des Distribution")
##    col1, col2 = st.beta_columns(2)
##    is_age_selected = col1.radio("Distribution Age ",('non','oui'))
##    is_incomdis_selected = col2.radio('Distribution Revenus ',('non','oui'))
##
##    return is_age_selected,is_incomdis_selected 
##
##def age_distribution():
##    df = pd.DataFrame({'Age':data['DAYS_BIRTH'],
##                'Solvabilite':data['TARGET']})
##
##    dic = {0: "solvable", 1: "non solvable"}        
##    df=df.replace({"Solvabilite": dic})    
##      
##    #fig = ff.create_distplot([revenus_solvable],['solvable'] ,bin_size=.25)
##    fig = plt.histogram(df,x="Age", color="Solvabilite", nbins=40)
##    st.subheader("Distribution des ages selon la sovabilité")
##    st.plotly_chart(fig)
##
##
##def revenu_distribution():
##    df = pd.DataFrame({'Revenus':data['AMT_INCOME_TOTAL'],
##                'Solvabilite':data['TARGET']})
##
##    dic = {0: "solvable", 1: "non solvable"}        
##    df=df.replace({"Solvabilite": dic})    
##      
##    #fig = ff.create_distplot([revenus_solvable],['solvable'] ,bin_size=.25)
##    fig = px.histogram(df,x="Revenus", color="Solvabilite", nbins=40)
##    st.subheader("Distribution des revenus selon la sovabilité")
##    st.plotly_chart(fig)


#--------------------------- Client Predection --------------------------
##
##def show_client_predection(data):
##    client_id = st.number_input("Donnez Id du Client",100002)
##    if st.button('Voir Client'):
##        client=data[data['SK_ID_CURR']==client_id]
##        #display_client_info(str(client['SK_ID_CURR'].values[0]),str(client['AMT_INCOME_TOTAL'].values[0]),str(round(client['DAYS_BIRTH'].values[0])),str(round(client['DAYS_EMPLOYED']/-365).values[0]))   
##        #st.header('ID :'+str(client['SK_ID_CURR'][0]))
##        #st.write(data['age_bins'].value_counts())
##        y_pred,y_proba = predict_client_par_ID("lgbm",client_id)
##        st.info('Prediction du client : '+str(int(100*y_proba[0][0]))+' %')
##        client_prediction= st.progress(0)
##        for percent_complete in range(int(100*client['pred_prob'].values[0])):
##            time.sleep(0.01)
##
##        client_prediction.progress(percent_complete + 1)
##        if(client['pred_prob'].values[0]<seuil_risque):
##            st.success('Client solvable')
##        if(client['pred_prob'].values[0]>=seuil_risque):
##            st.error('Client non solvable')
##
##        st.subheader("Tous les détails du client :")
##        st.write(client)
##        
##
##    
##        
##        #Bar Chart
##        age_bins = data['age_bins'].value_counts(sort=False)
##
##        d = {'Ages par Decennie': age_bins.index, 'Nombre de clients par Decennie':age_bins.values}
##        ages_decinnie = pd.DataFrame(data=d)
##
##        ages_decinnie['Ages par Decennie'] = ages_decinnie['Ages par Decennie'].astype(str)
##        idx_decinnie = ages_decinnie[ages_decinnie['Ages par Decennie'] == client['age_bins'].values[0]].index
##
##        colors = ['lightslategray',] * len(ages_decinnie['Nombre de clients par Decennie'])
##        colors[idx_decinnie.values[0]] = 'crimson'
##
##        fig = go.Figure(data=[go.Bar(
##            x=ages_decinnie['Ages par Decennie'],
##            y=ages_decinnie['Nombre de clients par Decennie'],
##            marker_color=colors # marker color can be a single color value or an iterable
##        )])
##        fig.update_layout(title_text='Nombre de Clients par Décinnie')
##
##        st.plotly_chart(fig)
##
##        #Line Chart
##        fig = go.Figure()
##        fig.add_trace(go.Scatter(y=data['pred_prob'], x=data['AMT_INCOME_TOTAL'],mode='markers', name='Revenus des autres clients'))
##        fig.add_trace(go.Scatter(y=client['pred_prob'], x=client['AMT_INCOME_TOTAL'],mode='markers', name='Revenu de ce Client',marker=dict(size=[25])))
##        st.plotly_chart(fig)
##
##
##
##
##
##
###--------------------------- model analysis -------------------------
##### Confusion matrixe
##def matrix_confusion (X,y):
##    cm = confusion_matrix(X, y)
##    print('\nTrue Positives(TP) = ', cm[0,0])
##    print('\nTrue Negatives(TN) = ', cm[1,1])
##    print('\nFalse Positives(FP) = ', cm[0,1])
##    print('\nFalse Negatives(FN) = ', cm[1,0])
##    return  cm
##
##def show_model_analysis():
##    conf_mtx = matrix_confusion (y_pred_test_export['y_test'],y_pred_test_export['y_predicted'])
##    #st.write(conf_mtx)
##    fig = go.Figure(data=go.Heatmap(
##                   z=conf_mtx,
##                    x=[ 'Actual Negative:0','Actual Positive:1'],
##                   y=['Predict Negative:0','Predict Positive:1'],
##                   hoverongaps = False))
##    st.plotly_chart(fig)
##
##    fpr, tpr, thresholds = roc_curve(y_pred_test_export['y_test'],y_pred_test_export['y_probability'])
##
##    fig = px.area(
##        x=fpr, y=tpr,
##        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
##        labels=dict(x='False Positive Rate', y='True Positive Rate'),
##        width=700, height=500
##    )
##    fig.add_shape(
##        type='line', line=dict(dash='dash'),
##        x0=0, x1=1, y0=0, y1=1
##    )
##
##    fig.update_yaxes(scaleanchor="x", scaleratio=1)
##    fig.update_xaxes(constrain='domain')
##    st.plotly_chart(fig)
##
##### ----------------------- Prédiction d'un client ----------------
##
##def file_selector(folder_path='.'):
##    filenames = os.listdir(folder_path)
##    selected_filename = st.selectbox('Selectionner un fichier client', filenames)
##    return os.path.join(folder_path, selected_filename)
##
##def show_client_prediction():
##    st.subheader("Selectionner source des données du client")
##    selected_choice = st.radio("",('Client existant dans le dataset','Nouveau client'))
##
##    if selected_choice == 'Client existant dans le dataset':
##        client_id = st.number_input("Donnez Id du Client",100002)
##        if st.button('Prédire Client'):
##            y_pred,y_proba = predict_client_par_ID("lgbm",client_id)
##            st.info('Probabilité de solvabilité du client : '+str(100*y_proba[0][0])+' %')
##            st.info("Notez que 100% => Client non slovable ")
##
##            if(y_proba[0][0]<seuil_risque):
##                st.success('Client prédit comme solvable')
##            if(y_proba[0][0]>=seuil_risque):
##                st.error('Client prédit comme non solvable !')
##
##    if selected_choice == 'Nouveau client':   
##        filename = file_selector()
##        st.write('Fichier du nouveau client selectionné `%s`' % filename)
##        
##        if st.button('Prédire Client'):
##            nouveau_client = pd.read_csv(filename)
##            y_pred,y_proba = predict_client("lgbm",nouveau_client)
##            st.info('Probabilité de solvabilité du client : '+str(100*y_proba[0][0])+' %')
##            st.info("Notez que 100% => Client non slovable ")
##            
##            if(y_proba[0][0]<seuil_risque):
##                st.success('Client prédit comme solvable')
##            if(y_proba[0][0]>=seuil_risque):
##                st.error('Client prédit comme non solvable !')


def main():
    API_URL = "http://127.0.0.1:5000/api/"
# LIST OF SK_ID_CURR
    # Get list of SK_IDS (cached)
    def get_sk_id_list():
        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"
        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of SK_IDS from the content
        SK_IDS = content['data']

        return SK_IDS
    
    SK_IDS = get_sk_id_list()

    # Logo "Prêt à dépenser"
    image = Image.open('logo.png')
    st.sidebar.image(image, width=280)
    st.title('Tableau de bord - "Prêt à dépenser"')

    # Load the data
#--------------
#
# LOAD DATA
    data = pd.read_csv("data/df_final.csv", nrows = 10000)  
    #DAYS_CREDIT = data['DAYS_CREDIT'] 
    #print (data.head())
    y_pred_test_export = pd.read_csv("data/y_pred_test_export.csv",nrows = 10000)
    #Preparation des données age
    data['DAYS_BIRTH']= data['DAYS_BIRTH']/-365
    bins= [0,10,20,30,40,50,60,70,80]
    data['age_bins'] = (pd.cut(data['DAYS_BIRTH'], bins=bins)).astype(str)

    train_set = pd.read_csv('data/application_train.csv',nrows = 1000)
    #X_train = pd.read_csv("data/X_train_transformed.csv",nrows=1000)
    #X_test = pd.read_csv("data/X_test_transformed.csv",nrows=1000)
    #train_set = pd.read_csv('data/application_train.csv',nrows=sample_size)
    #train_set = pd.read_csv("data/X_train_transformed.csv",nrows=1000)
    print (train_set.head())
    #################################################
    # LIST OF SK_ID_CURR
    #Get list of SK_IDS (cached)
    ### Title
    st.title('Home Credit Default Risk')

    ### Sidebar
    st.sidebar.title("Menus")
    sidebar_selection = st.sidebar.radio(
        'Select Menu:',
        ['Overview', 'Data Analysis', 'Model & Prediction', 'Prédire solvabilité client'],
    )

    if sidebar_selection == 'Overview':
        selected_item =""
        with st.spinner('Data load in progress...'):
            time.sleep(2)
        st.success('Data loaded')
        show_data(data) 
        #show_overview(data)   

    if sidebar_selection == 'Data Analysis':
        selected_item = st.sidebar.selectbox('Select Menu:', 
                                    ('Graphs', 'Distributions'))

    if sidebar_selection == 'Model & Prediction':
        selected_item = st.sidebar.selectbox('Select Menu:', 
                                        ( 'Prediction','Model'))

    if sidebar_selection == 'Prédire solvabilité client':
        selected_item="predire_client"

    seuil_risque = st.sidebar.slider("Seuil de Solvabilité", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    if selected_item == 'Data':
        show_data(data)  

    #if selected_item == 'Solvency':
        #show_overview (data)
        
    if selected_item == 'Graphs':
        #hist_graph()
        is_educ_selected,is_statut_selected,is_income_selected = filter_graphs()
        if(is_educ_selected=="oui"):
            education_type(train_set)
        if(is_statut_selected=="oui"):
            statut_plot()
        if(is_income_selected=="oui"):  
            income_type()

    if selected_item == 'Distributions':
        is_age_selected,is_incomdis_selected = filter_distribution()
        if(is_age_selected=="oui"):
            age_distribution()
        if(is_incomdis_selected=="oui"):
            revenu_distribution()     

    if selected_item == 'Prediction':
        show_client_predection(data)

    if selected_item == 'Model':
        show_model_analysis(data)

    if selected_item == 'predire_client':
        show_client_prediction(data)


    ##################################################
    # Selecting applicant ID
    select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', SK_IDS, key=1)
    st.write('You selected: ', select_sk_id)


#################################################################

# Ajout de données à data

    if not data.empty:
        st.write(data.head(10))
    else:
        st.write("Le DataFrame est vide.")    

def lecture_X_test_original():
    X_test_original = pd.read_csv("Data/X_test_original.csv")
    X_test_original = X_test_original.rename(columns=str.lower)
    return X_test_original

def lecture_X_test_clean():
    X_test_clean = pd.read_csv("Data/X_test_clean.csv")
    #st.dataframe(X_test_clean)
    return X_test_clean

def lecture_description_variables():
    description_variables = pd.read_csv("Data/description_variable.csv", sep=";")
    return description_variables


if __name__ == "__main__":

    lecture_X_test_original()
    lecture_X_test_clean()
    lecture_description_variables()

    # Titre 1
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                1. Quel est le score de votre client ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    ##########################################################
    # Création et affichage du sélecteur du numéro de client #
    ##########################################################
    liste_clients = list(lecture_X_test_original()['sk_id_curr'])
    col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
    with col1:
        ID_client = st.selectbox("*Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant 👇*", 
                                (liste_clients))
        st.write("Vous avez sélectionné l'identifiant n° :", ID_client)
    with col2:
        st.write("")

    #################################################
    # Lecture du modèle de prédiction et des scores #
    #################################################
    model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
    y_pred_lgbm = model_LGBM.predict(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))    # Prédiction de la classe 0 ou 1
    y_pred_lgbm_proba = model_LGBM.predict_proba(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1)) # Prédiction du % de risque

    # Récupération du score du client
    y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
    y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],
                                    lecture_X_test_clean()['sk_id_curr']], axis=1)
    #st.dataframe(y_pred_lgbm_proba_df)
    score = y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['sk_id_curr']==ID_client]
    score_value = round(score.proba_classe_1.iloc[0]*100, 2)

    # Récupération de la décision
    y_pred_lgbm_df = pd.DataFrame(y_pred_lgbm, columns=['prediction'])
    y_pred_lgbm_df = pd.concat([y_pred_lgbm_df, lecture_X_test_clean()['sk_id_curr']], axis=1)
    y_pred_lgbm_df['client'] = np.where(y_pred_lgbm_df.prediction == 1, "non solvable", "solvable")
    y_pred_lgbm_df['decision'] = np.where(y_pred_lgbm_df.prediction == 1, "refuser", "accorder")
    solvabilite = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "client"].values
    decision = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "decision"].values

    ##############################################################
    # Affichage du score et du graphique de gauge sur 2 colonnes #
    ##############################################################
    col1, col2 = st.columns(2)
    with col2:
        st.markdown(""" <br> <br> """, unsafe_allow_html=True)
        st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1f}%**.")
        st.write(f"**Il y a donc un risque de {score_value:.1f}% que le client ait des difficultés de paiement.**")
        st.write(f"Le client est donc considéré par *'Prêt à dépenser'* comme **{solvabilite[0]}** \
                et décide de lui **{decision[0]}** le crédit. ")
    # Impression du graphique jauge
    with col1:
        fig = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = float(score_value),
                        mode = "gauge+number+delta",
                        title = {'text': "Score du client", 'font': {'size': 24}},
                        delta = {'reference': 35.2, 'increasing': {'color': "#3b203e"}},
                        gauge = {'axis': {'range': [None, 100],
                                'tickwidth': 3,
                                'tickcolor': 'darkblue'},
                                'bar': {'color': 'white', 'thickness' : 0.3},
                                'bgcolor': 'white',
                                'borderwidth': 1,
                                'bordercolor': 'gray',
                                'steps': [{'range': [0, 20], 'color': '#e8af92'},
                                        {'range': [20, 40], 'color': '#db6e59'},
                                        {'range': [40, 60], 'color': '#b43058'},
                                        {'range': [60, 80], 'color': '#772b58'},
                                        {'range': [80, 100], 'color': '#3b203e'}],
                                'threshold': {'line': {'color': 'white', 'width': 8},
                                            'thickness': 0.8,
                                            'value': 35.2 }}))

        fig.update_layout(paper_bgcolor='white',
                        height=400, width=500,
                        font={'color': '#772b58', 'family': 'Roboto Condensed'},
                        margin=dict(l=30, r=30, b=5, t=5))
        st.plotly_chart(fig, use_container_width=True)

    ################################
    # Explication de la prédiction #
    ################################
    # Titre 2
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                2. Comment le score de votre client est-il calculé ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    # Calcul des valeurs Shap
    explainer_shap = shap.TreeExplainer(model_LGBM)
    shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))

    # récupération de l'index correspondant à l'identifiant du client
    idx = int(lecture_X_test_clean()[lecture_X_test_clean()['sk_id_curr']==ID_client].index[0])

    # Graphique force_plot
    st.write("Le graphique suivant appelé `force-plot` permet de voir où se place la prédiction (f(x)) par rapport à la `base value`.") 
    st.write("Nous observons également quelles sont les variables qui augmentent la probabilité du client d'être \
            en défaut de paiement (en rouge) et celles qui la diminuent (en bleu), ainsi que l’amplitude de cet impact.")
    st_shap(shap.force_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            link='logit',
                            figsize=(20, 8),
                            ordering_keys=True,
                            text_rotation=0,
                            contribution_threshold=0.05))
    # Graphique decision_plot
    st.write("Le graphique ci-dessous appelé `decision_plot` est une autre manière de comprendre la prédiction.\
            Comme pour le graphique précédent, il met en évidence l’amplitude et la nature de l’impact de chaque variable \
            avec sa quantification ainsi que leur ordre d’importance. Mais surtout il permet d'observer \
            “la trajectoire” prise par la prédiction du client pour chacune des valeurs des variables affichées. ")
    st.write("Seules les 15 variables explicatives les plus importantes sont affichées par ordre décroissant.")
    st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            feature_names=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list(),
                            feature_order='importance',
                            feature_display_range=slice(None, -16, -1), # affichage des 15 variables les + importantes
                            link='logit'))

    # Titre 3
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                3. Lexique des variables </h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    st.write("La base de données globale contient un peu plus de 200 variables explicatives. Certaines d'entre elles étaient peu \
            renseignées ou peu voir non disciminantes et d'autres très corrélées (2 variables corrélées entre elles \
            apportent la même information : l'une d'elles est donc redondante).")
    st.write("Après leur analyse, 56 variables se sont avérées pertinentes pour prédire si le client aura ou non des difficultés de paiement.")

    pd.set_option('display.max_colwidth', None)
    st.dataframe(lecture_description_variables())



def test_load():
    assert load_dataset(1).size == 5000  
