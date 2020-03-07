import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering


class Labeller():

    def __init__(self, X):
        self.X = X
        self.n_clusters = 5

    def get_cluster(self):
        train_data = self.X
        train_data = train_data[[x for x in train_data if x not in ['transit', 'error']]]

        # Create proportional feature
        train_data['depth_prop'] = train_data['depth'] / train_data['depth'].values[0]

        # Get Cumulative Differences
        train_data['depth_prop_diff'] = train_data['depth_prop'] - train_data['depth_prop'].shift(-3)

        # Create 5-period sma (simple moving average)
        train_data['depth_prop_sma3'] = train_data['depth_prop'].rolling(window=3).mean()
        train_data['depth_prop_sma7'] = train_data['depth_prop'].rolling(window=7).mean()
        train_data['depth_prop_sma3_slope'] = train_data['depth_prop'].rolling(window=3).mean() - train_data['depth_prop'].rolling(window=3).mean().shift(-1)
        slope_vars = {'level': [-0.3, 0.3],
                      'neg': [float('-inf'), -0.3],
                      'pos': [0.3, float('inf')]}
        for key in slope_vars:
            train_data['is_{}'.format(key)] = train_data.apply(lambda x: 1 if x['depth_prop_sma3_slope'] > slope_vars[key][0] and x['depth_prop_sma3_slope'] <= slope_vars[key][1] else 0, axis=1)

        # print(train_data['depth_prop'].shape)

        # Apply kmeans
        from sklearn.cluster import KMeans

        # Select non-nan
        mask = ~train_data.isna()
        mask = mask.all(axis=1)

        train_data = train_data.loc[mask]

        # MaxMin Scaling between 0->1
        X = train_data
        norm = MinMaxScaler()
        X = norm.fit(X).transform(X)

        # Applying Unsupervised Learning for Labels
        kmeans = AgglomerativeClustering(n_clusters=self.n_clusters).fit(X)

        # Dropping nans from features
        if kmeans.labels_.shape[0] != train_data.shape[0]:
            train_data = train_data.loc[mask]
            kmeans.labels_ = kmeans.labels_[mask]
            kmeans.labels_ = kmeans.labels_[mask]

        # Settings Labels as color
        train_data['labels_int'] = kmeans.labels_
        color_map = {'0': 'red',
                     '1': 'blue',
                     '2': 'green',
                     '3': 'gray',
                     '4': 'yellow',
                     '5': 'purple'}
        train_data['color'] = train_data.apply(lambda x: color_map[str(int(x['labels_int']))], axis=1)

        # Create Readable Labels
        label_map = {'0': 'Bottom',
                     '1': 'Upward Slope',
                     '2': 'Post Transit',
                     '3': 'Downward Slope',
                     '4': 'Pre Transit',
                     '5': 'Other'}
        train_data['labels_desc'] = train_data.apply(lambda x: label_map[str(int(x['labels_int']))], axis=1)

        cmap = plt.get_cmap("magma")
        cmap = plt.get_cmap("YlOrBr")
        sns.scatterplot(x=range(len(train_data['depth'])),
                        y=train_data['depth'] * -1, hue=train_data['labels_int'])

        plt.show()
        # print(X)

        return train_data
