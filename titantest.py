import streamlit as st
import pandas as pd
import numpy as np 
import time
import matplotlib.pyplot as plt

#讀data
# SibSp = 手足 siblings / 配偶 spouses
# Parch = 父母 parents / 小孩 children
#http://bit.ly/kaggletrain

@st.cache(allow_output_mutation=True)
def load():
	df=pd.read_csv("C:\\Users\\user80917\\Desktop\\train.csv")
	return df
df=load()

dtrain=df.filter(regex='Survived|Age|SibSp|Parch|Fare|Sex|Pclass')

	
from sklearn.ensemble import RandomForestRegressor

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
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
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
	df.loc[(df.Sex==str("female")) & (df.Parch>=1) ,'Mother']=1
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

#對age和fare標準化至(-1,1)
from sklearn.preprocessing import StandardScaler
StandardScaler=StandardScaler()

def scaler(df):
	
	age_p=StandardScaler.fit(df[['Age']].values.reshape(-1,1))
	df['Age_sc']=StandardScaler.fit_transform(df[['Age']].values.reshape(-1,1),age_p)

	fare_p=StandardScaler.fit(df[['Fare']].values.reshape(-1,1))
	df['Fare_sc']=StandardScaler.fit_transform(df[['Fare']].values.reshape(-1,1),fare_p)
	return df

dtrain=scaler(dtrain)


from sklearn.ensemble import RandomForestClassifier

train_df=dtrain.filter(regex='Survived|Age_sc|SibSp|Parch|Familysize|Fare_sc|Sex_.*|Pclass_.*|Child|Mother')


#y即Survival結果
y = train_df['Survived'].values

#X即features屬性值
X = train_df.iloc[:, 1:].values

rfr1 = RandomForestClassifier(n_estimators=2000,criterion='gini',min_samples_split=12,min_samples_leaf=1,random_state=1,n_jobs=-1) 
							 
rfr1.fit(X,y)


#streamlit
st.title("Titanic Test")
'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.01)

'...and now we\'re done!'

zero_list=(0,0,0,0,0,0,0,0,0,0)
test_df=pd.DataFrame([zero_list],columns=['Pclass_1','Pclass_2','Pclass_3','Fare','Sex','Age','SibSp','Parch','Child','Mother'])

#pclass/fare,pclass=1&2&3最便宜票價(換算)=£2100 & £1900 & £600，資料中平均票價為£84/£21/£14
fare_option=st.selectbox(' Welcome to Titanic ! Choose your class and start journey. ',
('Economy Class-£600','Bussiness Class-£1900','First Class-£2100'))

'You selected:', fare_option

if (fare_option=='First Class-£2100'):
	test_df.loc[0,'Pclass_1']=1
	test_df.loc[0,'Pclass_2']=0
	test_df.loc[0,'Pclass_3']=0
	test_df.loc[0,'Fare']=84
elif (fare_option=='Bussiness Class-£1900'):
	test_df.loc[0,'Pclass_2']=1
	test_df.loc[0,'Pclass_1']=0
	test_df.loc[0,'Pclass_3']=0
	test_df.loc[0,'Fare']=21
else:
	test_df.loc[0,'Pclass_3']=1
	test_df.loc[0,'Pclass_1']=0
	test_df.loc[0,'Pclass_2']=0
	test_df.loc[0,'Fare']=14
	
#gender
gender_option = st.selectbox('Please select your gender.',('Male','Female'))

'You selected:', gender_option

if (gender_option=='Male'):
	test_df['Sex_male']=1
	test_df['Sex_female']=0
else:
	test_df['Sex_male']=0
	test_df['Sex_female']=1

#Age
Age = st.slider('How old are you?', 0, 100, 1)
st.write("I'm ", Age, 'years old')

test_df.loc[0,'Age']=Age

#SibSp
spouse_option = st.selectbox('Are you married?',('No','Yes'))

if (spouse_option=='Yes'):
	spouse=1
else:
	spouse=0

sibling = st.number_input('How many sibling do you have?',value=0,max_value=20,step=1)
SibSp=spouse+sibling
test_df.loc[0,'SibSp']=SibSp

#Parch
children=st.number_input('How many children do you have?',value=0,max_value=20,step=1)
st.write('The current number is ', children)

parent_option=st.selectbox('Do you want to bring your parents?',('No','Yes'))

if (parent_option=='Yes'):
	parent=2
	
if (parent_option=='No'):
	parent_num=st.selectbox('Choose the number of parents that you would like to bring.',(' Ok , I will ask one of them. - 1' , 'No , I do NOT play with them! - 0' ))
	
	if (parent_num==' Ok , I will ask one of them. - 1'):
		parent=1
	else:
		parent=0

Parch=children+parent
test_df.loc[0,'Parch']=Parch

test_df=preprocessing(test_df)

#fit原資料，再transform test_df的age
age_p=StandardScaler.fit(dtrain[['Age']].values)
test_df['Age_sc']=StandardScaler.transform(test_df[['Age']].values.reshape(-1,1),age_p)

fare_p=StandardScaler.fit(dtrain[['Fare']].values)
test_df['Fare_sc']=StandardScaler.transform(test_df[['Fare']].values.reshape(-1,1),fare_p)

test_df.drop(['Fare','Age','Sex'],axis=1,inplace=True)

st.write(test_df)

prediction=rfr1.predict(test_df)
prob=rfr1.predict_proba(test_df)
prob_df=pd.DataFrame(prob,columns=['Dead','Survived'])
dead_prob=prob_df.loc[0,'Dead']
survived_prob=prob_df.loc[0,'Survived']


#顯示生存機率
if st.checkbox(' Show the Result ! '):
	st.subheader('Result')
	st.write('Your Survivability is %0.2f , Mortality is %0.2f'%(survived_prob,dead_prob))

