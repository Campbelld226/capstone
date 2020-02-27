from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
iris = datasets.load_iris()


data=pd.DataFrame({
    'Transit Depth':iris.data[:,0],
    'Transit Length':iris.data[:,1],
    'Depth Derivative':iris.data[:,2],
    'Second Derivative Depth':iris.data[:,3],
    'Transits':iris.target
})

data.head()

X = data[['Transit Depth', 'Transit Length', 'Depth Derivative', 'Second Derivative Depth']]
Y = data['Transits']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)


clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


clf.fit(X_train, Y_train)
feature_imp = pd.Series(clf.feature_importances_,index=['Transit Depth', 'Transit Length', 'Depth Derivative', 'Second Derivative Depth']).sort_values(ascending=False)
print(feature_imp)
print("Transit Predicition: ", clf.predict([[3,5,4,2]]), "\n[1]:Transit\n[2]:No Transit\n[3]:Unsure")

'exec(%matplotlib inline)'

sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
























#Y_pred = clf.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


#print(clf.predict([[3,5,4,2]]))

