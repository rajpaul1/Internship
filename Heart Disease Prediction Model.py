#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("E:\data\heart.csv")


# In[3]:


data


# In[4]:


data.info()


# In[5]:


sns.countplot(x="target", data = data, palette="magma")


# In[6]:



sns.countplot(x="sex", data = data, palette="bwr")


# In[7]:


pd.crosstab(data.age,data.target).plot(kind="bar",figsize=(17,6))


# In[8]:


sns.countplot(x="fbs", data = data, palette="bwr")


# In[9]:


plt.scatter(data.age[data.target==0],data.thalach[data.target==0])
plt.scatter(data.age[data.target==1],data.thalach[data.target==1])
plt.legend(["0","1"])
plt.xlabel("Age")
plt.ylabel("thalach")
plt.show()


# In[10]:


plt.scatter(data.age[data.target==0],data.chol[data.target==0])
plt.scatter(data.age[data.target==1],data.chol[data.target==1])
plt.legend(["0","1"])
plt.xlabel("Age")
plt.ylabel("chol")
plt.show()


# In[11]:



from sklearn.preprocessing import StandardScaler


# In[12]:



sc = StandardScaler()
data[["age","trestbps","chol","thalach","oldpeak"]] = sc.fit_transform(data[["age","trestbps","chol","thalach","oldpeak"]])


# In[13]:


data


# In[14]:



data[["sex","cp","fbs","restecg","exang","slope","ca","thal"]] = data[["sex","cp","fbs","restecg","exang","slope","ca","thal"]].astype(object)


# In[15]:


data=pd.get_dummies(data)


# In[16]:


X = data.copy()
y = data["target"]


# In[17]:


X.drop(columns="target",axis=1,inplace=True)


# In[18]:



X = X.values
y=y.values


# In[20]:


X


# In[21]:



y


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=1)


# In[24]:


X_train.shape


# In[25]:


from sklearn.metrics import classification_report


# In[26]:


from sklearn.neighbors import KNeighborsClassifier


# In[27]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[28]:


knn.fit(X_train,y_train)


# In[29]:


pred = knn.predict(X_test)


# In[30]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[59]:


cm1 = confusion_matrix(y_test,pred)
cm1


# In[56]:


ac1=accuracy_score(y_test, pred)
ac1


# In[33]:


print(classification_report(pred,y_test))


# In[34]:


from sklearn.svm import SVC


# In[35]:


svc = SVC()


# In[36]:


svc.fit(X_train,y_train)


# In[37]:


pred2 = svc.predict(X_test)


# In[38]:



cm2=confusion_matrix(y_test,pred2)
cm2


# In[39]:


ac2=accuracy_score(y_test, pred2)
ac2


# In[40]:



print(classification_report(pred2,y_test))


# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


rfc = RandomForestClassifier()


# In[43]:


rfc.fit(X_train,y_train)


# In[44]:


pred3 = rfc.predict(X_test)


# In[45]:


cm3=confusion_matrix(y_test,pred3)
cm3


# In[46]:


ac3=accuracy_score(y_test, pred3)
ac3


# In[47]:


print(classification_report(pred3,y_test))


# In[48]:


from sklearn.naive_bayes import GaussianNB


# In[49]:


nb = GaussianNB()


# In[50]:


nb.fit(X_train,y_train)


# In[51]:



pred4 = nb.predict(X_test)


# In[52]:


cm4=confusion_matrix(y_test,pred4)
cm4


# In[53]:


ac4=accuracy_score(y_test, pred4)
ac4


# In[54]:


print(classification_report(pred4,y_test))


# In[57]:


print("Accuracy of models")

plt.bar(0,ac)
plt.bar(1,ac1)
plt.bar(2,ac2)
plt.bar(3,ac3)
plt.bar(4,ac4)

plt.xticks([0,1,2,3,4], ['KNN','DTC','RFC','SVC','NB'])
plt.show()


# In[60]:


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrices",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm1,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Random forest classifier Confusion Matrix")
sns.heatmap(cm2,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm3,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm4,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.show()


# In[ ]:




