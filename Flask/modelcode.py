import pickle
import pandas as pd

df = pd.read_csv('Crop_recommendation.csv')

X = df.drop('label', axis=1)
y = df['label']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=48)


from sklearn.preprocessing import StandardScaler
sc  = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
my_model = RandomForestClassifier(n_estimators=75,criterion='entropy',random_state=1)
result = my_model.fit(X_train,y_train)

#predictions = result.predict(X_test)

pickle.dump(my_model, open("model.pkl", "wb"))