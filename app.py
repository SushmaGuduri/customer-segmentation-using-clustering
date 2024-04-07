
from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore") 

filename = 'final_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
df = pd.read_csv("Clustered_Customer_Data.csv")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)
st.title("Prediction")

with st.form("my_form"):
    Education=st.number_input(label='Education',step=1)
    Marital_Status=st.number_input(label='Marital_Status',step=1)
    Income=st.number_input(label='Income',step=1)
    Kidhome=st.number_input(label='Kidhome',step=1)
    Teenhome=st.number_input(label='Teenhome',step=1)
    Recency=st.number_input(label='Recency',step=1)
    MntWines=st.number_input(label='MntWines',step=1)
    MntFruits=st.number_input(label='MntFruits',step=1)
    MntMeatProducts=st.number_input(label='MntMeatProducts',step=1)
    MntFishProducts=st.number_input(label='MntFishProducts',step=1)
    MntSweetProducts=st.number_input(label='MntSweetProducts',step=1)
    MntGoldProds=st.number_input(label='MntGoldProds',step=1)
    NumDealsPurchases=st.number_input(label='NumDealsPurchases',step=1)
    NumWebPurchases=st.number_input(label='NumWebPurchases',step=1)
    NumCatalogPurchases=st.number_input(label='NumCatalogPurchases',step=1)
    NumStorePurchases=st.number_input(label='NumStorePurchases',step=1)
    NumWebVisitsMonth=st.number_input(label='NumWebVisitsMonth',step=1)
    AcceptedCmp3=st.number_input(label='AcceptedCmp3',step=1)
    AcceptedCmp4=st.number_input(label='AcceptedCmp4',step=1)
    AcceptedCmp5=st.number_input(label='AcceptedCmp5',step=1)
    AcceptedCmp1=st.number_input(label='AcceptedCmp1',step=1)
    AcceptedCmp2=st.number_input(label='AcceptedCmp2',step=1)
    Complain=st.number_input(label='Complain',step=1)
    Response=st.number_input(label='Response',step=1)
    TotalAmountSpent=st.number_input(label='TotalAmountSpent',step=1)
    Year_Customer=st.number_input(label='Year_Customer',step=1)
    TotalPromotionsAccepted=st.number_input(label='TotalPromotionsAccepted',step=1)
    TotalChildren=st.number_input(label='TotalChildren',step=1)
    
    data=[[Education,Marital_Status,Income,Kidhome,Teenhome,Recency,MntWines,MntFruits,MntMeatProducts,MntFishProducts,MntSweetProducts,MntGoldProds,NumDealsPurchases,NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumWebVisitsMonth,AcceptedCmp3,AcceptedCmp4,AcceptedCmp5,AcceptedCmp1,AcceptedCmp2,Complain,Response,TotalAmountSpent,Year_Customer,TotalPromotionsAccepted,TotalChildren]]

    submitted = st.form_submit_button("Submit")

if submitted:
    clust=loaded_model.predict(data)[0]
    print('Data Belongs to Cluster',clust)

    cluster_df1=df[df['Label']==clust]
    plt.rcParams["figure.figsize"] = (20,3)
    # for c in cluster_df1.drop(['Label'],axis=1):
    #     fig, ax = plt.subplots()
    #     grid= sns.FacetGrid(cluster_df1, col='Label')
    #     grid= grid.map(plt.hist, c)
    #     plt.show()
    #     st.pyplot(figsize=(5, 5))


