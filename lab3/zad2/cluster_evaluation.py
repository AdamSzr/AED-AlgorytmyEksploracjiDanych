
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import numpy as np


def cluster_evaluation(X, range_n_clusters, xlowerlim=-0.1, store_values=False):
    '''Returns Silhouette plot for clusterwise samples and a second plot to show  
    distribution of points in the feature space of first two attributes

                   X : a dataframe.
    range_n_clusters : a list of candidate values for the number of clusters.
          xlower_lim : the lower limit of x axis for the silhouette scores plot
          Usually, The silhouette coefficient can range from -1, 1.
        store_values : Bool. False
        If True, function will return a dictionary storing the silhouette scores
        at overall level and individual clusters.

    '''
    # Create a subplot with 1 row and 2 columns
    # rowcount = len(range_n_clusters)
    # fig, axr = plt.subplots(nrows = rowcount, ncols = 2)
    # fig.set_size_inches(18, rowcount*3)

    # Intercluster gap for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    intercluster_gap = 10

    # Initializing a dictionary
    dict_Kcluster = {}

    for i, n_clusters in enumerate(range_n_clusters):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 4)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but depending on the
        # observed minimum silhouette coefficient, we will set the lower limit of x.
        ax1.set_xlim([xlowerlim, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * intercluster_gap])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        # Store values - silhoutte score in dictionary
        if store_values == True:
            dict_Kcluster[n_clusters] = {}
            dict_Kcluster[n_clusters]['silhouette_score'] = {}
            dict_Kcluster[n_clusters]['silhouette_score']['size'] = \
                int(sample_silhouette_values.size)
            dict_Kcluster[n_clusters]['silhouette_score']['avg'] = \
                silhouette_avg.round(6)
            dict_Kcluster[n_clusters]['silhouette_score']['max'] = \
                sample_silhouette_values.max().round(6)
            dict_Kcluster[n_clusters]['silhouette_score']['min'] = \
                sample_silhouette_values.max().round(6)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg.round(6))

        y_lower = intercluster_gap
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]

            # Store values - cluster sizes in dictionary
            if store_values == True:
                dict_Kcluster[n_clusters]['cluster'+str(i)] = {}
                dict_Kcluster[n_clusters]['cluster'+str(i)]['size'] = \
                    int(size_cluster_i)
                dict_Kcluster[n_clusters]['cluster'+str(i)]['avg'] = \
                    ith_cluster_silhouette_values.mean()
                dict_Kcluster[n_clusters]['cluster'+str(i)]['max'] = \
                    ith_cluster_silhouette_values.max()
                dict_Kcluster[n_clusters]['cluster'+str(i)]['min'] = \
                    ith_cluster_silhouette_values.min()

            # print(f'   Cluster {i}: {size_cluster_i}')

            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)

            # Plotting silhouette values corresponding to each value in sample
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # ---------------------------------------------------------------------------
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
    plt.show()

    # Return dictionary if store_values is True
    if store_values == True:
        return dict_Kcluster
