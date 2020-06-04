import streamlit as st
import pandas as pd
import numpy as np 
import time
import module.titanic as modu
from PIL import Image
from sklearn.preprocessing import StandardScaler


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

image = Image.open('titanic.jpg')
st.image(image, caption='Titanic',use_column_width=True)

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
	test_df['spouse']=1
else:
	test_df['spouse']=0

sibling = st.number_input('How many sibling do you have?',value=0,max_value=20,step=1)
SibSp=test_df['spouse']+sibling

#children
children=st.number_input('How many children do you have?',value=0,max_value=20,step=1)
st.write('The current number is ', children)

#parents
parent_option=st.selectbox('Do you want to bring your parents?',('No','Yes'))

if (parent_option=='Yes'):
	test_df['parent']=2
	
if (parent_option=='No'):
	parent_num=st.selectbox('Choose the number of parents that you would like to bring.',(' Ok , I will ask one of them. - 1' , 'No , I do NOT play with them! - 0' ))
	
	if (parent_num==' Ok , I will ask one of them. - 1'):
		test_df['parent']=1
	else:
		test_df['parent']=0

test_df=modu.preprocessing(test_df)

#parch
test_df['Parch']=children+test_df['parent']

#mother
if (test_df.iloc[0]['Sex_female']==1) & (test_df.iloc[0]['Parch']>0):
	test_df['Mother']=int(1)

#familysize	
test_df['Familysize']=test_df['Parch']+test_df['SibSp']

test_df.drop(['spouse','Sex','parent'],axis=1,inplace=True)

st.write("Please check your data.")
st.write(test_df)

prob=modu.makeprediction(test_df)

#顯示生存機率
if st.checkbox(' Show the Result ! '):
	st.subheader('Result')
	st.write('Your Survivability is %0.2f , Mortality is %0.2f'%(prob[1],prob[0]))
