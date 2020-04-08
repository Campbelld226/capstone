import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans


class Labeler:

    def __init__(self, X):
        self.X = X
        self.n_clusters = 5
    
    def cluster_method(self, val):
        # MaxMin Scaling between 0->1
        norm = MinMaxScaler()
        X = norm.fit(self.X).transform(self.X)

        if val == 1:
            # Applying Unsupervised Learning for Labels
            clustering = AgglomerativeClustering(n_clusters=self.n_clusters).fit(X)
        
        elif val == 2:
            clustering = SpectralClustering(n_clusters=self.n_clusters).fit(X)

        elif val == 3:
            clustering = KMeans(n_clusters=self.n_clusters).fit(X)

        # Dropping nans from features
        mask = ~self.X.isna()
        mask = mask.all(axis=1)
        if clustering.labels_.shape[0] != self.X.shape[0]:
            self.X = self.X.loc[mask]
            clustering.labels_ = clustering.labels_[mask]
            clustering.labels_ = clustering.labels_[mask]

        return clustering

    def order_clusters(self, clustering):
        train_data = self.X
        color_array = ['red', 'blue', 'green', 'yellow', 'gray']
        labels_desc_array = ['Exterior Ingress', 'Interior Ingress', 'Greatest Transit', 'Interior Egress', 'Exterior Egress']
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
        self.X = train_data



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
        self.X = train_data

        # agglom_clustering = self.cluster_method(1)
        # spectral_clustering = self.cluster_method(2)
        # k_means = self.cluster_method(3)
        # train_data = self.display(mask, agglom_clustering, "Agglomerative Clustering")
        # train_data = self.display(mask, spectral_clustering, "Spectral Clustering")
        # train_data = self.display(mask, k_means, "K-Means Clustering")

    def average(self):
        train_data = self.X
        cluster0 = []
        cluster1 = [] 
        cluster2 = []
        cluster3 = []
        cluster4 = []
        total = mean(train_data['depth'])
        for index, label in enumerate(train_data['depth']):
            if list(train_data['labels_int'])[index] == 0:
                cluster0.append(label)
            elif list(train_data['labels_int'])[index] == 1:
                cluster1.append(label)
            elif list(train_data['labels_int'])[index] == 2:
                cluster2.append(label)
            elif list(train_data['labels_int'])[index] == 3:
                cluster3.append(label)
            elif list(train_data['labels_int'])[index] == 4:
                cluster4.append(label) 
        
        overall_avg = [(mean(cluster0)/total)*100, (mean(cluster1)/total)*100,
                    (mean(cluster2)/total)*100, (mean(cluster3)/total)*100,
                    (mean(cluster4)/total)*100]
        return overall_avg
        #all_avg.append(total)


if __name__ == '__main__':
    l = Labeler(pd.read_csv('test/CoRoT-2b_1.2.csv'))
    l.get_data()
    agglom_clustering = l.cluster_method(1)
    spectral_clustering = l.cluster_method(2)
    k_means = l.cluster_method(3)

    l.order_clusters(agglom_clustering)
    sns.scatterplot(x=range(len(l.X['depth'])),
                    y=l.X['depth'] * -1, hue=l.X['labels_desc'])
    # plt.title(cluster_name)
    plt.show()
    l.order_clusters(spectral_clustering)
    sns.scatterplot(x=range(len(l.X['depth'])),
                    y=l.X['depth'] * -1, hue=l.X['labels_desc'])
    # plt.title(cluster_name)
    plt.show()
    l.order_clusters(k_means)
    sns.scatterplot(x=range(len(l.X['depth'])),
                    y=l.X['depth'] * -1, hue=l.X['labels_desc'])
    # plt.title(cluster_name)
    plt.show()

    print("Transit: ", l.average())
    
    # l2 = Labeler(pd.read_csv('test/no_transit.csv'))
    # l2.get_data()
    # print("No Transit: ", l2.average())


# Create fake data
# import decimal
# time_array = []
# depth_array = []
# val = decimal.Decimal(2458000.0)
# while val < 2458000.2:
#     time_array.append(float(val))
#     depth_array.append(round(random.uniform(-0.001, 0.001), 6))
#     val += decimal.Decimal(0.001)
# test = pd.DataFrame({'epoch': time_array, 'depth': depth_array})
# test['epoch'] = time_array
# test['depth'] = depth_array
# test.to_csv()