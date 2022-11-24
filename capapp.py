import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
my_dataset="Telco-Customer-Churn_21para.csv"
def explore_data(dataset):
    df=pd.read_csv(dataset)
    return df
data=explore_data(my_dataset)
features_n=19
st.set_page_config(initial_sidebar_state="collapsed")
html_temp = """
    <div style="background-color:grey;padding:10px">
    <h2 style="color:white;text-align:center;">Customer Attrition Prediction</h2>
    </div>"""
st.markdown(html_temp,unsafe_allow_html=True)

# background
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://th.bing.com/th/id/R.856f31d9f475501c7552c97dbe727319?rik=Eq9oehb4QunXVw&riu=http%3a%2f%2fwww.baltana.com%2ffiles%2fwallpapers-5%2fWhite-Background-High-Definition-Wallpaper-16573.jpg&ehk=I38kgsJb2jc3ycTK304df0ig%2flhB3PaaXRrqcPVwDgA%3d&risl=&pid=ImgRaw&r=0")
 }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    activities=["EDA","Visualize Data","Prediction","Model statistics","About"]
    choice=st.sidebar.selectbox("",activities)
    if choice=="EDA":
        EDA()
    if choice=="Visualize Data":
        DataVisualization()
    if choice=="Prediction":
        prediction()
    if choice=="Model statistics":
        statistics()
    if choice=="About":
        about()
def EDA():
    st.header("Exploratory Data Analysis")
    method_names=["Show dataset","Head","Tail","Shape","Describe","Missing value","Columns Names"]
    method_operation=[data,data.head(),data.tail(),data.shape,data.describe(),data.isnull().sum(),data.columns]

    for i in range(len(method_names)):
        if st.checkbox(method_names[i]):
            st.write(method_operation[i])
    all_columns=list(data.columns)
    if st.checkbox("Select columns to show"):
        selected_columns=st.multiselect("Select column",all_columns)
        new_df=data[selected_columns]
        st.dataframe(new_df)

def DataVisualization():
    st.header("Data Visualization")
    if st.checkbox("Numerical variable"):
        column_name=st.selectbox("",("Select column","MonthlyCharges","SeniorCitizen","TotalCharges","Age"))
        if column_name=="MonthlyCharges":
            plt.figure(figsize=(5,3))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(sns.distplot(data[column_name]))
            st.pyplot()
        elif column_name=="SeniorCitizen":
            plt.figure(figsize=(5,3))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(sns.distplot(data[column_name]))
            st.pyplot()
        elif column_name=="TotalCharges":
            plt.figure(figsize=(4,3))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
            st.write(sns.distplot(data[column_name]))
            st.pyplot()
    if st.checkbox("Categorical variable"):
        column_name=st.selectbox("",("Select column","Contract","PhoneService","tenure","InternetService"))
        if column_name=="Contract":
            plt.figure(figsize=(5,3))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
        elif column_name=="PhoneService":
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
        elif column_name=="tenure":
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
        elif column_name=="InternetService":
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.write(plt.hist(data[column_name],color="skyblue",edgecolor="black"))
            st.pyplot()
def prediction():
    data = pd.read_csv("Telco-Customer-Churn_21para.csv")
    data.head()
    data = data.drop(["customerID"], axis = 1)
    data.drop(labels=data[data["TotalCharges"] == ' '].index,axis=0,inplace=True)
    data = data.reset_index(drop = True)
    data['TotalCharges'] = pd.to_numeric(data.TotalCharges, errors='coerce')
    data.SeniorCitizen = data.SeniorCitizen.map({0: "No", 1: "Yes"})
    cat=[]
    num=[]
    for i in data.columns:
        if(data[i].dtype=="object"):
            cat.append(i)
        else:
            num.append(i)
    data['Churn'] = data['Churn'].astype('object')
    y=data['Churn']
    # y=y.astype('int')
    cat.remove('Churn')
    data_encoded=data.drop('Churn',axis=1)
    data_encoded.head()
    cat_data = pd.get_dummies(data_encoded[cat])
    scaler = StandardScaler()
    features = data_encoded[num]
    features = scaler.fit_transform(features)
    num_features=pd.DataFrame(features)
    attributes = pd.concat([cat_data, num_features], axis=1)
    X=attributes
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 1)
    st.header("Prediction")
    col_list=[0]*features_n

    gen_col=["Female","Male"]
    gen_num=list(range(len(gen_col)))
    gen=st.radio("Select gender",gen_num,format_func=lambda x:gen_col[x])
    col_list[0]=gen  

    age_col=[0,1]
    age_num=list(range(len(age_col)))
    age=st.radio("Senior Citizen?",age_num,format_func=lambda x:age_col[x])
    col_list[1]=age

    partner_col=["Yes","No"]
    partner_num=list(range(len(partner_col)))
    partner=st.radio("Partner?",partner_num,format_func=lambda x:partner_col[x])
    col_list[2]=partner

    dpdt_col=["Yes","No"]
    dpdt_num=list(range(len(dpdt_col)))
    dpdt=st.radio("Dependant?",dpdt_num,format_func=lambda x:dpdt_col[x])
    col_list[3]=dpdt

    ps_col=["Yes","No"]
    ps_num=list(range(len(ps_col)))
    ps=st.radio("Phone Service?",ps_num,format_func=lambda x:ps_col[x])
    col_list[4]=ps

    ml_col=["Yes","No"]
    ml_num=list(range(len(ml_col)))
    ml=st.radio("Multiple Lines?",ml_num,format_func=lambda x:ml_col[x])
    col_list[5]=ml

    # InternetService=["InternetService","Fiber Optic","DSL","No internet service"]
    # InternetS_option=st.selectbox("",InternetService)
    # if InternetS_option=="Fiber Optic":
    #     col_list[7]=InternetS_option
    #     # col_list[10]=0
    # elif InternetS_option=="DSL":
    #     col_list[7]=InternetS_option
    #     # col_list[10]=1
    # elif InternetS_option=="No internet service":
    #     col_list[7]=InternetS_option
        # col_list[10]=0
    InternetService=["Fiber Optic","DSL","No internet service"]
    IS=list(range(len(InternetService)))
    ise=st.radio("Choose Your Internet Service Type:",IS,format_func=lambda x:InternetService[x])
    col_list[6]=ise
    
    os_col=["Yes","No"]
    os_num=list(range(len(os_col)))
    os=st.radio("Online Security?",os_num,format_func=lambda x:os_col[x])
    col_list[7]=os

    ob_col=["Yes","No"]
    ob_num=list(range(len(ob_col)))
    ob=st.radio("Online Backup?",ob_num,format_func=lambda x:ob_col[x])
    col_list[8]=ob

    dp_col=["Yes","No"]
    dp_num=list(range(len(dp_col)))
    dp=st.radio("Device Protection?",dp_num,format_func=lambda x:dp_col[x])
    col_list[9]=dp

    ts_col=["Yes","No"]
    ts_num=list(range(len(ts_col)))
    ts=st.radio("Tech Support?",ts_num,format_func=lambda x:ts_col[x])
    col_list[10]=ts

    stv_col=["Yes","No"]
    stv_num=list(range(len(stv_col)))
    stv=st.radio("Streaming TV?",stv_num,format_func=lambda x:stv_col[x])
    col_list[11]=stv

    sm_col=["Yes","No"]
    sm_num=list(range(len(sm_col)))
    sm=st.radio("Streaming Movie?",sm_num,format_func=lambda x:sm_col[x])
    col_list[12]=sm

    Contract=["One year","month-to-month"]
    con_num=list(range(len(Contract)))
    con=st.radio("Choose your Contract Type?",con_num,format_func=lambda x:Contract[x])
    col_list[13]=con

    pbl_col=["Yes","No"]
    pbl_num=list(range(len(pbl_col)))
    pbl=st.radio("Paper Billing?",pbl_num,format_func=lambda x:sm_col[x])
    col_list[14]=pbl

    paymentmethod=["Electronic check","Mail check","Credit card (automatic)","Bank transfer (automatic)"]
    pm_option=list(range(len(paymentmethod)))
    pm=st.radio("Payment Method?",pm_option,format_func=lambda x:paymentmethod[x])
    col_list[15]=pm

    Tenure_col=st.slider("Enter tenure : ",0,72)
    col_list[16]=Tenure_col

    Monthlycharges=st.number_input("Enter Monthly charges: ",step=0.01)
    col_list[17]=Monthlycharges

    Totalcharges=st.number_input("Enter Total charges:",step=0.01)
    col_list[18]=Totalcharges


    st.write(col_list)
    if st.checkbox("Your entries"):
        d={}
        feature=["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]
        for i in range(len(feature)):
            if i<19:
                d[feature[i]]=col_list[i]
            else:
                d[feature[i]]=[col_list[i],col_list[i+1]]
        st.write(d)

    if st.button("Predict"):
        original=[]
        if(col_list[0]==0):
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(1)
        if(col_list[1]==0):
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(1)

        if(col_list[2]==0):
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(1)

        if(col_list[3]==0):
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(1)

        if(col_list[4]==0):
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(1)

        if(col_list[5]==0):
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[5]==1):
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(1)

        if(col_list[6]==0):
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[6]==1):
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(1)

        if(col_list[7]==0):
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[7]==1):
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(1)

        if(col_list[8]==0):
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[8]==1):
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(1)

        if(col_list[9]==0):
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[9]==1):
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(1)

        if(col_list[10]==0):
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[10]==1):
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(1)

        if(col_list[11]==0):
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[11]==1):
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(1)

        if(col_list[12]==0):
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[12]==1):
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(1)

        if(col_list[13]==0):
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[13]==1):
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(1)

        if(col_list[14]==0):
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(1)
        # 'PaymentMethod_Bank transfer (automatic)','PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check','PaymentMethod_Mailed check',

        if(col_list[15]==0):
            original.append(1)
            original.append(0)
            original.append(0)
            original.append(0)
        elif(col_list[15]==1):
            original.append(0)
            original.append(1)
            original.append(0)
            original.append(0)
        elif(col_list[15]==2):
            original.append(0)
            original.append(0)
            original.append(1)
            original.append(0)
        else:
            original.append(0)
            original.append(0)
            original.append(0)
            original.append(1)
        original.append(col_list[16])
        original.append(col_list[17])
        original.append(col_list[18])
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train, y_train)
        # original2.reshape(1,-1)
        # scaler = MinMaxScaler()
        y_pred = classifier.predict([original])
        st.write(y_pred)
def statistics():
    pass
def about():
    st.write("This is the project that is done by M. Vandith Reddy, KVS. Siddhartha, K Deepak Srinivas")
if __name__=="__main__":
    main() 