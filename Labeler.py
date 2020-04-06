import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans


class Labeler:

    def __init__(self, X):
        self.X = X
        self.n_clusters = 5
    
    def cluster_method(self, X, val):

        if val == 1:
            # Applying Unsupervised Learning for Labels
            clustering = AgglomerativeClustering(n_clusters=self.n_clusters).fit(X)
        
        elif val == 2:
            clustering = SpectralClustering(n_clusters=self.n_clusters).fit(X)

        elif val == 3:
            clustering = KMeans(n_clusters=self.n_clusters).fit(X)

        return clustering

    def display(self, mask, X, clustering, cluster_name, train_data):
        # Dropping nans from features
        if clustering.labels_.shape[0] != train_data.shape[0]:
            train_data = train_data.loc[mask]
            clustering.labels_ = clustering.labels_[mask]
            clustering.labels_ = clustering.labels_[mask]

        color_array = ['red', 'blue', 'green', 'yellow', 'gray']
        labels_desc_array = ['Pre-Transit', 'Downward Slope', 'Bottom', 'Upward Slope', 'Post-Transit']
        cluster_dict = {}
        cluster_num = -1
        labels_int = []
        color = []
        labels_desc = []
        for index, label in enumerate(clustering.labels_):
            if label not in cluster_dict.keys():
                cluster_num = cluster_num + 1
                cluster_dict[label] = cluster_num

            labels_int.append(cluster_dict[label])
            color.append(color_array[cluster_dict[label]])
            labels_desc.append(labels_desc_array[cluster_num])

        train_data['labels_int'] = labels_int
        train_data['labels_desc'] = labels_desc
        train_data['color'] = color


        # # Settings Labels as color
        # train_data['labels_int'] = clustering.labels_
        # print(clustering.labels_)
        # #print(train_data['labels_int'])
        # #print(train_data['labels_int'])
        # color_map = {'0': 'red',
        #             '1': 'blue',
        #             '2': 'green',
        #             '3': 'gray',
        #             '4': 'yellow',
        #             '5': 'purple'}
        # train_data['color'] = train_data.apply(lambda x: color_map[str(int(x['labels_int']))], axis=1)
        #
        # # Create Readable Labels
        # label_map = {'0': 'Bottom',
        #            '1': 'Upward Slope',
        #            '2': 'Post Transit',
        #            '3': 'Downward Slope',
        #           '4': 'Pre Transit',
        #           '5': 'Other'}
        # train_data['labels_desc'] = train_data.apply(lambda x: label_map[str(int(x['labels_int']))], axis=1)

        sns.scatterplot(x=range(len(train_data['depth'])),
                        y=train_data['depth'] * -1, hue=train_data['labels_desc'])
        plt.title(cluster_name)

        plt.show()
        return train_data

    def get_data(self):
        train_data = self.X
        # TODO: Convert a bunch of files to the proper format, with no transit value
        #  Maybe leave error in the files for now and we can filter from here
        train_data = train_data[[x for x in train_data if x not in ['error']]]

        # Create proportional feature
        train_data['depth_prop'] = train_data['depth'] / train_data['depth'].values[0]

        # Get cumulative differential
        train_data['depth_prop_diff'] = train_data['depth_prop'] - train_data['depth_prop'].shift(-3)

        # Create 3-period and 7-period sma (simple moving average)
        train_data['depth_prop_sma3'] = train_data['depth_prop'].rolling(window=3).mean()
        train_data['depth_prop_sma7'] = train_data['depth_prop'].rolling(window=7).mean()

        # Create 3-period sma differential
        train_data['depth_prop_sma3_slope'] = train_data['depth_prop'].rolling(window=3).mean() - train_data['depth_prop'].rolling(window=3).mean().shift(-1)

        # Create 3-period sma differential slope type feature (positive, negative, or neutral slope)
        slope_vars = {'level': [-0.3, 0.3],
                    'neg': [float('-inf'), -0.3],
                    'pos': [0.3, float('inf')]}
        for key in slope_vars:
            train_data['is_{}'.format(key)] = train_data.apply(lambda x: 1 if x['depth_prop_sma3_slope'] > slope_vars[key][0] and x['depth_prop_sma3_slope'] <= slope_vars[key][1] else 0, axis=1)

        # Select non-nan
        mask = ~train_data.isna()
        mask = mask.all(axis=1)
        train_data = train_data.loc[mask]

        # MaxMin Scaling between 0->1
        X = train_data
        norm = MinMaxScaler()
        X = norm.fit(X).transform(X)

        agglom_clustering = self.cluster_method(X, 1)
        spectral_clustering = self.cluster_method(X, 2)
        k_means = self.cluster_method(X, 3)
        train_data = self.display(mask, X, agglom_clustering, "Agglomerative Clustering", train_data)
        train_data = self.display(mask, X, spectral_clustering, "Spectral Clustering", train_data)
        train_data = self.display(mask, X, k_means, "K-Means Clustering", train_data)
        return train_data


# if __name__ == '__main__':
#     l = Labeler(pd.read_csv('test/CoRoT-2b_1.2.csv'))
#     l.get_data()