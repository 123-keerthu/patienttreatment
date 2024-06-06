from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def result(request):
    if request.method == 'POST':
        hmt = float(request.POST.get('hmt'))
        hmg = float(request.POST.get('hmg'))
        ert = float(request.POST.get('ert'))
        leu = float(request.POST.get('leu'))
        trm = float(request.POST.get('trm'))
        mch = float(request.POST.get('mch'))
        mchc = float(request.POST.get('mchc'))
        mcv = float(request.POST.get('mcv'))
        age = float(request.POST.get('age'))
        sex = request.POST.get('sex')
        sex = 0 if sex == 'M' else 1
        path = "C:\\Users\\KEERTHANA  S\\Desktop\\AIML internship 23-24\\Keerthana\\disease\\data.csv"
        data = pd.read_csv(path)
        data.isna().any()
        data_n=pd.DataFrame(data)
        data_n.dropna(inplace=True)
        # data_n['SEX'] = data_n['SEX'].map({'M':0,'F':1})
        n_sex=LabelEncoder()
        data_n['SEX']=n_sex.fit_transform(data_n['SEX'])
 
        inputs = data_n[['HAEMATOCRIT','HAEMOGLOBINS','ERYTHROCYTE','LEUCOCYTE','THROMBOCYTE','MCH','MCHC','MCV','AGE','SEX']]
        output = data_n['SOURCE']
        re = RandomForestClassifier(n_estimators=100,random_state=42)
        re.fit(inputs,output)
        pred = re.predict([[hmt,hmg,ert,leu,trm,mch,mchc,mcv,age,sex]])
        return render(request,"user.html",{'prediction':pred[0]})
    else:
        return render(request,"user.html")







