from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: Label some of our data
#  [start_of_transit end_of_transit max_depth baseline start_time end_time]
#  [part_of_transit derivative_of_point peak tangent ]

# TODO: Use our real data
iris = datasets.load_iris()
# print(iris)

#Two dimensional data in the format columns:rows
data=pd.DataFrame({
    'Transit Depth':iris.data[:,0],
    'Transit Length':iris.data[:,1],
    'Depth Derivative':iris.data[:,2],
    'Second Derivative Depth':iris.data[:,3],
    'Transits':iris.target
})

print(iris.data)
#X and Y components for our classifier, X being the features, Y being the labels
X = data[['Transit Depth', 'Transit Length', 'Depth Derivative', 'Second Derivative Depth']]
Y = data['Transits']

#Splitting the data into testing and training data, giving our program 70% training data and 30% testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

#
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

#fitting the classifier to our data, and extracting the feature importance.
clf.fit(X_train, Y_train)
feature_imp = pd.Series(clf.feature_importances_,index=['Transit Depth', 'Transit Length', 'Depth Derivative', 'Second Derivative Depth']).sort_values(ascending=False)
Y_pred = clf.predict(X_test)


print(feature_imp)
print("Transit Predicitions: ", Y_pred, "\n[0]:Transit\n[1]:No Transit\n[2]:Unsure")
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

# TODO: Why isn't this working?
#exec(%matplotlib inline)

sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
























#Y_pred = clf.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


#print(clf.predict([[3,5,4,2]]))

