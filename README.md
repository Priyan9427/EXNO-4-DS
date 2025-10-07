# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

# Developed By: PRIYAN U

# Register no: 212224040254

```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```

<img width="1426" height="878" alt="image" src="https://github.com/user-attachments/assets/e10d473c-72cc-4c2f-96b6-ec4e44fba911" />

```
 data.isnull().sum()
```

<img width="360" height="655" alt="image" src="https://github.com/user-attachments/assets/175757e2-0dfd-4c83-ab8b-2f53b2570305" />

```
 missing=data[data.isnull().any(axis=1)]
 missing
```

<img width="1415" height="867" alt="image" src="https://github.com/user-attachments/assets/e3df2e34-dcf3-486a-a729-253e0fa08c16" />

```
 data2=data.dropna(axis=0)
 data2
```

<img width="1438" height="897" alt="image" src="https://github.com/user-attachments/assets/d84e8ed1-56b7-4f89-b380-c47500f9fd0c" />

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

<img width="1430" height="530" alt="image" src="https://github.com/user-attachments/assets/ab27049c-6c75-4dba-bddf-b59f998bb6c7" />

```
data2
```

<img width="1422" height="806" alt="image" src="https://github.com/user-attachments/assets/fcc5c732-c3b8-4b0b-b348-8247a90e4e96" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

<img width="1431" height="691" alt="image" src="https://github.com/user-attachments/assets/48972ea4-3ef4-4f80-9bee-fcf609cdd531" />

```
columns_list=list(new_data.columns)
print(columns_list)
```

<img width="1420" height="126" alt="image" src="https://github.com/user-attachments/assets/b98c92b0-37b6-4bc9-9af4-20cb6fcecf7a" />

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

<img width="1437" height="136" alt="image" src="https://github.com/user-attachments/assets/4a5fa9fd-8982-4c36-9bdb-c7bbfb6d8ef6" />

```
y=new_data['SalStat'].values
print(y)
```

<img width="396" height="101" alt="image" src="https://github.com/user-attachments/assets/48fe0b7d-22bb-4615-9c66-5d3a56b2d291" />


```
x=new_data[features].values
print(x)
```

<img width="499" height="248" alt="image" src="https://github.com/user-attachments/assets/1c8bd196-fc8c-45e9-8542-b427bd340060" />


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

<img width="853" height="201" alt="image" src="https://github.com/user-attachments/assets/e98b460a-7bbc-4fb3-8c4b-334576a4803b" />


```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

<img width="611" height="163" alt="image" src="https://github.com/user-attachments/assets/6db17275-435e-4015-b871-d287853e6452" />


```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

<img width="531" height="113" alt="image" src="https://github.com/user-attachments/assets/e3183980-24c0-4f54-92a5-f6a378bba7b7" />


```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

<img width="693" height="86" alt="image" src="https://github.com/user-attachments/assets/0a95a406-3bb1-4f19-92b8-f5b2a7609c42" />

```
data.shape
```

<img width="205" height="97" alt="image" src="https://github.com/user-attachments/assets/cd9eb1cb-7ba7-4d8a-9e2b-5fe030f2c84b" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
'Feature1': [1,2,3,4,5],
'Feature2': ['A','B','C','A','B'],
'Feature3': [0,1,1,0,1],
'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="1379" height="557" alt="image" src="https://github.com/user-attachments/assets/b5973976-4271-475c-b6e8-5794d22e9c5b" />


```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

<img width="674" height="457" alt="image" src="https://github.com/user-attachments/assets/9146f863-3668-4e98-b57b-0937b86c0b81" />


```
tips.time.unique()
```

<img width="540" height="118" alt="image" src="https://github.com/user-attachments/assets/53b4a27c-cd25-4eac-99e7-685239491012" />


```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

<img width="649" height="170" alt="image" src="https://github.com/user-attachments/assets/accae3f7-341c-4a7d-b03d-94a4247ab9d7" />


```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

<img width="540" height="152" alt="image" src="https://github.com/user-attachments/assets/15ee5bb3-804a-4dfc-be33-f18bfe7ce87a" />



# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process andsave the data to a file is been executed.
