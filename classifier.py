import inline as inline
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: What other features should we be using?
#  - Derivatives?
# TODO: Create a script to convert files to the proper csv format
#  - Add the labels at the top
#  - Remove unnecessary data
#  - Replace whitespace with commas
# TODO: Label more data
# TODO: Make the learning persistent so we don't need to train each time
#  - Use pickling
# TODO: Train on a lot more data
# TODO: Figure out a good way to show the light curve graphs so they're not upside down
#  - All depth values get larger as the transit occurs, but some are negative and some
#       are positive, making some graphs go upside down. Learn matplotlib better
# TODO: Figure out a good graphical representation of the predicted data
#  - Can we graph the raw points and maybe plot the predicted transit over top?
# TODO: Try predicting with Cartledge's shitty data
# TODO: Clean the git repository
#  - There are many branches on branches on branches. Time to merge and clean up
# TODO: Build a requirements file for pip
# TODO: Figure out how to package this all in a convenient way for people to use
# TODO: Build a GUI if we have time
# gradient, derivative, local avg, covolution, neighbourhood points, peaks, distance from mean of signal, abs min and max
# fourier transform -> operation -> inverse fourier
# polynomial features
# look at the

# iris = datasets.load_iris()
# print(iris)
train_data = pd.read_csv('test/CoRoT-2b_1.6.txt')
print('TRAIN DATA:\n', train_data)
test_data = pd.read_csv('test/CoRoT-2b_1.7.txt')
print('TEST DATA:\n', test_data)

# Two dimensional data in the format columns:rows
# data=pd.DataFrame({
#     'Transit Depth':iris.data[:,0],
#     'Transit Length':iris.data[:,1],
#     'Depth Derivative':iris.data[:,2],
#     'Second Derivative Depth':iris.data[:,3],
#     'Transits':iris.target
# })
# print(iris.data)


# X and Y components for our classifier, X being the features, Y being the labels
# X = data[['Transit Depth', 'Transit Length', 'Depth Derivative', 'Second Derivative Depth']]
# Y = data['Transits']
train_y = train_data['transit']
train_x = train_data.drop('transit', axis=1)
test_y = test_data['transit']
test_x = test_data.drop('transit', axis=1)


# Splitting the data into testing and training data, giving our program 70% training data and 30% testing data
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

#
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                             max_depth=None, max_features='auto', max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=1, min_samples_split=2,
                             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                             oob_score=False, random_state=None, verbose=0,
                             warm_start=False)

# fitting the classifier to our data, and extracting the feature importance.
clf.fit(train_x, train_y)
feature_imp = pd.Series(clf.feature_importances_, index=['epoch', 'depth', 'error']).sort_values(ascending=False)
pred_y = clf.predict(test_x)


print(feature_imp)
print("Transit Predictions: ", pred_y, "\n[0]:Transit\n[1]:No Transit")
print("Accuracy:", metrics.accuracy_score(test_y, pred_y))

# TODO: Why isn't this working?
# exec(%matplotlib inline)

sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
