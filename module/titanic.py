import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

#讀data
# SibSp = 手足 siblings / 配偶 spouses
# Parch = 父母 parents / 小孩 children

@st.cache(allow_output_mutation=True)
def load():
	df=pd.read_csv("train.csv")
	return df
df=load()

dtrain=df.filter(regex='Survived|Age|SibSp|Parch|Fare|Sex|Pclass')
	
### 使用 RandomForestRegressor 填補缺失的年齡屬性

def set_missing_ages(df):
    
    # 把已有的數值型features取出来丟進RandomForestRegressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    
    # 乘客分成已知年齡和未知年齡兩部分
    known_age = age_df[age_df.Age.notnull()]
    unknown_age = age_df[age_df.Age.isnull()]
    
    # y即目標年齡
    y = known_age.iloc[:, 0]

    # X即feature屬性值
    X = known_age.iloc[:, 1:]
    
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=500, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型進行未知年齡結果預測
    predictedAges = rfr.predict(unknown_age.iloc[:, 1:])

    # 用得到的預測結果填補原缺失data
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    return df,rfr
	
dtrain, rfr = set_missing_ages(dtrain)

def preprocessing(df): 
	#familysize
	df['Familysize']=df['Parch']+df['SibSp']

	#child
	df["Child"] = df["Age"].apply(lambda x: 1 if x < 15 else 0)
	
	#Mother
	df['Mother']='0'# DataFrame增加一直行
	df.loc[(df['Sex']==str("female")) & (df['Parch']>=1) ,'Mother']=1
	return df
dtrain=preprocessing(dtrain)

#one hot encoding
def onehot(df):
	d_Sex=pd.get_dummies(df['Sex'],prefix='Sex')
	d_Pclass=pd.get_dummies(df['Pclass'],prefix='Pclass')

	#把處理後的新增回dtrain,df
	df=pd.concat([df,d_Sex,d_Pclass],axis=1)
	df.drop(['Sex','Pclass'],axis=1,inplace=True)
	return df
dtrain=onehot(dtrain)


train_df=dtrain.filter(regex='Survived|Age|SibSp|Parch|Familysize|Fare|Sex_.*|Pclass_.*|Child|Mother')


#y即Survival結果
y = train_df['Survived'].values

#X即features屬性值
X = train_df.iloc[:, 1:].values

rfr1 = RandomForestClassifier(n_estimators=500,criterion='gini',min_samples_split=12,min_samples_leaf=1,random_state=1,n_jobs=-1) 
							 
rfr1.fit(X,y)

#return a tuple
def makeprediction(df):
	prediction=rfr1.predict(df)
	prob=rfr1.predict_proba(df)
	prob_df=pd.DataFrame(prob,columns=['Dead','Survived'])
	dead_prob=prob_df.loc[0,'Dead']
	survived_prob=prob_df.loc[0,'Survived']
	return dead_prob,survived_prob