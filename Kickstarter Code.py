#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 02:40:28 2021

@author: saadtariq
"""

import pandas as pd
import numpy as np
#Importing dataset
kickstarter = pd.read_excel("Kickstarter.xlsx")


kickstarter.head()
kickstarter.isna().sum()
kickstarter = kickstarter.drop(['launch_to_state_change_days'], axis =1)
kickstarter.isna().sum()
kickstarter = kickstarter.dropna()
kickstarter.isna().sum()

kickstarter = kickstarter[(kickstarter['state']=='successful') | (kickstarter['state']=='failed')]
kickstarter.shape
dummy_state=pd.get_dummies(kickstarter.state, prefix="state")
kickstarter = kickstarter.join(dummy_state)

dummy_staff=pd.get_dummies(kickstarter.staff_pick, prefix="staff_pick")
dummy_category=pd.get_dummies(kickstarter.category, prefix="category")
dummy_country = pd.get_dummies(kickstarter.country, prefix="country")
dummy_weekday = pd.get_dummies(kickstarter.created_at_weekday, prefix="created_at_weekday")
kickstarter = kickstarter.join(dummy_staff)
kickstarter = kickstarter.join(dummy_category)
kickstarter = kickstarter.join(dummy_country)
kickstarter = kickstarter.join(dummy_weekday)

list(kickstarter.columns)

y = kickstarter["state_successful"]
x = kickstarter[[
 'goal',
 'backers_count',
 'static_usd_rate',
 'name_len',
 'name_len_clean',
 'blurb_len',
 'blurb_len_clean',
 'staff_pick_False',
 'staff_pick_True',
 'category_Academic',
 'category_Apps',
 'category_Blues',
 'category_Comedy',
 'category_Experimental',
 'category_Festivals',
 'category_Flight',
 'category_Gadgets',
 'category_Hardware',
 'category_Immersive',
 'category_Makerspaces',
 'category_Musical',
 'category_Places',
 'category_Plays',
 'category_Robots',
 'category_Shorts',
 'category_Software',
 'category_Sound',
 'category_Spaces',
 'category_Thrillers',
 'category_Wearables',
 'category_Web',
 'category_Webseries',
  'created_at_month',
 'created_at_day',
 'created_at_yr',
 'created_at_hr',
 'country_AT',
 'country_AU',
 'country_BE',
 'country_CA',
 'country_CH',
 'country_DE',
 'country_DK',
 'country_ES',
 'country_FR',
 'country_GB',
 'country_HK',
 'country_IE',
 'country_IT',
 'country_LU',
 'country_MX',
 'country_NL',
 'country_NO',
 'country_NZ',
 'country_SE',
 'country_SG',
 'country_US',
 'created_at_weekday_Friday',
 'created_at_weekday_Monday',
 'created_at_weekday_Saturday',
 'created_at_weekday_Sunday',
 'created_at_weekday_Thursday',
 'created_at_weekday_Tuesday',
 'created_at_weekday_Wednesday']]



#Running feature selection technique Random Forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0)
model = randomforest.fit(x, y)
model.feature_importances_

#Running feature selection technique RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=5000)
rfe = RFE(lr, n_features_to_select=6)
model = rfe.fit(x, y)
model.ranking_

lr = LogisticRegression(max_iter=5000)
rfe= RFE(lr,n_features_to_select=1)
model=rfe.fit(x,y)
model.support_

pd.DataFrame(list(zip(x.columns,model.ranking_)),columns=(['predictor','ranking']))

#Running feature selection technique LASSO
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(x)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)
model.fit(X_std,y)
model.coef_
test=pd.DataFrame(list(zip(x.columns,model.coef_)),columns=(['predictor','coefficient']))
for i in range(len(test)):
            if test['coefficient'][i]!=0.0:
                print(test['predictor'][i],'--',i) 
#Choosing optimal value of alpha
for  i in np.arange(0, 0.1, 0.01):
        model = Lasso(alpha=i)
        model.fit(X_std,y)
        print("*-----------------"*10)
        print(i)
        print(model.coef_)
        
        test=pd.DataFrame(list(zip(x.columns,model.coef_)),columns=(['predictor','coefficient']))
#need to remove the predictors which have '0' as coeff from the set
        for i in range(len(test)):
            if test['coefficient'][i]!=0.0:
                print(test['predictor'][i],'--',i)  
           

f1_scores = []
optimum_alpha = 0
max_f1_score = 0 
               
for i in np.arange(0, 0.1, 0.01):
    # Running Lasso to find Predictors
    
    model1 = Lasso(alpha = i)
    model1.fit(X_std, y)
    model1.coef_
    imp_matrix = pd.DataFrame(list(zip(x.columns,model1.coef_)), columns = ['predictor','coefficient'])
    final_matrix = imp_matrix[imp_matrix.coefficient != 0]
    predictor_list = final_matrix['predictor']
    predictor_list = predictor_list.to_numpy()
    
    # Update X

    X2 = kickstarter[predictor_list]

    # Split the data

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.3, random_state = 5)

    from sklearn.ensemble import GradientBoostingClassifier
    gbt = GradientBoostingClassifier(random_state = 0, min_samples_split = 2, n_estimators = 100)
    model2 = gbt.fit(X_train, y_train)
    y_test_pred1 = model2.predict(X_test)

    from sklearn import metrics

    from sklearn.metrics import accuracy_score
    print('Accuracy:','alpha =',i, accuracy_score(y_test, y_test_pred1))
    # Calculate the F1 score
    print('F1 Score:','alpha =', i, metrics.f1_score(y_test, y_test_pred1))
    
    f1_scores.append(metrics.f1_score(y_test, y_test_pred1))
    if metrics.f1_score(y_test, y_test_pred1) >= max_f1_score:
        max_f1_score = metrics.f1_score(y_test, y_test_pred1)
        optimum_alpha = i                

#Alpha = 0.01
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(x)
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)
model.fit(X_std,y)
model.coef_

test=pd.DataFrame(list(zip(x.columns,model.coef_)),columns=(['predictor','coefficient']))
#need to remove the predictors which have '0' as coeff from the set
for i in range(len(test)):
    if test['coefficient'][i]!=0.0:
        print("'",test['predictor'][i],"'",",") 
#Choosing alpha = 0.05 to get 6 predictors
#Checking correlation between predictors

import seaborn as sn
import matplotlib.pyplot as plt

df1 = pd.DataFrame(kickstarter,columns=['goal', 'backers_count', 'pledged', 'usd_pledged'
])

df = pd.DataFrame(kickstarter,columns=['goal', 'backers_count', 
'name_len_clean', 'name_len'
])

df.dtypes

corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot = True)
plt.show()

corrMatrix1 = df1.corr()
sn.heatmap(corrMatrix1, annot = True)
plt.show()

#Ran correlation matrix on continuous variables, we see that name_len and name_len_clean have high correlation
#Not only do backers_count, pledged and usd_pledged have high correlation, but they also
#occur after project is created


X1 = kickstarter[['goal',
'name_len_clean' ,
'staff_pick_False' ,
'category_Blues' ,
'category_Experimental' ,
'category_Festivals' ,
'category_Flight' ,
'category_Immersive' ,
'category_Musical' ,
'category_Places' ,
'category_Plays' ,
'category_Shorts' ,
'category_Software' ,
'category_Spaces' ,
'category_Web' ,
'created_at_yr' ,
'country_AU' ,
'country_GB' ,
'country_IT' ,
'country_US' ,]]
y1 = kickstarter['state_successful']



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size=0.30,random_state=4)

# Run Logistic Regression (base model)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter = 100000)

model = lr.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

# Calculate Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_test_pred))
#Accuracy of 67.2%


##Random Forest (Model improvement)

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
model = randomforest.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_test_pred))

from sklearn import metrics
print('Precision Score:', metrics.precision_score(y_test, y_test_pred1))
print('Recall Score:', metrics.recall_score(y_test, y_test_pred))

# Calculate the F1 score
print('F1 Score:', metrics.f1_score(y_test, y_test_pred))

#Accuracy improved to 74% and F1 score improved to 59%

## Further Model improvement with GBT (FINAL MODEL)

from sklearn.ensemble import GradientBoostingClassifier

gbt = GradientBoostingClassifier(random_state = 0)
model1 = gbt.fit(X_train, y_train)
y_test_pred1 = model1.predict(X_test)

from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_test_pred1))

from sklearn import metrics
print('Precision Score:', metrics.precision_score(y_test, y_test_pred1))
print('Recall Score:', metrics.recall_score(y_test, y_test_pred1))

# Calculate the F1 score
print('F1 Score:', metrics.f1_score(y_test, y_test_pred1))

#Accuracy improved to 77.5% and F1 score improved to 61.68%


from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(random_state = 0))
sel.fit(X_train, y_train)
selected_feat = X_train.columns[(sel.get_support())]
selected_feat


#### Part 2
#=============================================================================
#Kmeans cannot be used on categorical data
list(kickstarter.columns)
X2 = kickstarter[[ 'goal',
  'name_len',
  'name_len_clean',
  'blurb_len',
  'blurb_len_clean',
  'deadline_month',
  'deadline_day',
  'deadline_yr',
  'deadline_hr',
  'state_changed_at_month',
  'state_changed_at_day',
  'state_changed_at_yr',
  'state_changed_at_hr',
  'created_at_month',
  'created_at_day',
  'created_at_yr',
  'created_at_hr',
  'launched_at_month',
  'launched_at_day',
  'launched_at_yr',
  'launched_at_hr',
  'create_to_launch_days',
  'launch_to_deadline_days']]
#=============================================================================
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.io as pio
import plotly.express as px
from pandas.plotting import *



df=X2

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    kmeanVal = KMeans(n_clusters=k)
    kmeanVal = kmeanVal.fit(df)
    Sum_of_squared_distances.append(kmeanVal.inertia_)
labels =kmeanVal.labels_
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

silhouette = silhouette_samples(df, labels)

print('\nSilhouette Score:',silhouette_score(df, labels))


#Taken only numerical columns
cols=['goal',
  'name_len_clean',
  'created_at_yr']
scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
pca = PCA(n_components=2)
df = pca.fit_transform(standardized_x)
reduced_df = pd.DataFrame(df, columns=['PC1','PC2'])
plt.scatter(reduced_df['PC1'], reduced_df['PC2'], alpha=.1, color='blue')
plt.xlabel('Principal Component Analysis 1')
plt.ylabel('Principal Component Analysis  2')
plt.show()

kmeans=KMeans(n_clusters=4)
model=kmeans.fit(reduced_df)
labels=model.predict(reduced_df)
reduced_df['cluster'] = labels
list_labels=labels.tolist()
count1=0
count2=0
count3=0
count4=0
for i in list_labels:
    if i==0:
        count1=count1+1
    elif i==1:
        count2=count2+1
    elif i==2:
        count3=count3+1
    elif i==3:
        count4=count4+1
u_labels=np.unique(labels)
print("\nTotal datapoints in cluster 1 (K Means):", count1)
print("Total datapoints in cluster 2 (K Means):", count2)
print("Total datapoints in cluster 3 (K Means):", count3)
print("Total datapoints in cluster 4 (K Means):", count4)
for i in u_labels:
    plt.scatter(df[labels == i , 0] , df[labels == i , 1] )
plt.legend(u_labels)
plt.show()


df=X2
cols=['goal',
  'name_len_clean',
  'created_at_yr']
df=df[cols]

scaler=StandardScaler()
standardized_x=scaler.fit_transform(df)
standardized_x=pd.DataFrame(standardized_x,columns=df.columns)
df=standardized_x
kmeans=KMeans(n_clusters=4)
model=kmeans.fit(df)
labels=model.predict(df)
df['cluster'] = labels
list_labels=labels.tolist()
count1=count2=count3=count4=0
for i in list_labels:
    if i==0:
        count1+=1
    elif i==1:
        count2+=1
    elif i==2:
        count3+=1
    elif i==3:
        count4+=1
u_labels=np.unique(labels)
print("Total datapoints in cluster 1 (K-Means):", count1)
print("Total datapoints in cluster 2 (K-Means):", count2)
print("Total datapoints in cluster 3 (K-Means):", count3)
print("Total datapoints in cluster 4 (K-Means):", count4)

pio.renderers.default = 'browser'

centroids = pd.DataFrame(kmeans.cluster_centers_)
fig = px.parallel_coordinates(centroids,labels=df.columns,color=u_labels)
fig.show()


