import pickle
import numpy as np
import spacy
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import squareform
from nltk.stem.snowball import SnowballStemmer
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
import seaborn as sns

# apply stemming to the items in the counter
def save_stemmed_Counter(counter, output_filename):
    en_stem = SnowballStemmer("english")
    de_stem = SnowballStemmer("german")
    es_stem = SnowballStemmer('spanish')
    fi_stem = SnowballStemmer('finnish')
    stemmed_list = [(de_stem.stem(de), en_stem.stem(en), es_stem.stem(es), fi_stem.stem(fi)) for (de, en, es, fi) in
                    counter.keys()]
    stem_counter = Counter(stemmed_list)
    with open(output_filename, 'wb') as f:
        pickle.dump(Counter(stem_counter), f)

# calculate the difference between two quadruples
def num_different_items(tuple1, tuple2):
    different_items = set(tuple1).symmetric_difference(set(tuple2))
    return int(len(different_items)/2)

def create_distance_matrix(dictionary):
    distance_matrix = {}
    all_keys = list(dictionary.keys())
    for i in range(len(all_keys)):
        for j in range(i + 1, len(all_keys)):
            tuple1 = all_keys[i]
            tuple2 = all_keys[j]
            distance = num_different_items(tuple1, tuple2)
            distance_matrix[(tuple1, tuple2)] = distance
    distance_matrix = np.array(list(distance_matrix.values()))
    square_matrix = squareform(distance_matrix)
    return square_matrix

def reduce_dimension(square_matrix, filename = 'more_than_one_stemmed.pkl'):
    mds = MDS(n_components=2, normalized_stress=True, metric=False)
    X_transformed = mds.fit_transform(square_matrix)
    with open(filename, 'wb') as f:
        pickle.dump(X_transformed, f)

def eigenvalues(square_matrix):
    n = square_matrix.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    double_centered_matrix = -0.5 * J.dot(square_matrix).dot(J)

    eigenvalues = np.linalg.eigvalsh(double_centered_matrix)

    k = 10
    top_k_eigenvalues = eigenvalues[-k:]

    print(top_k_eigenvalues)

def cluster_silhouette_score(transformed):
    k_values = range(2, 11)

    # Initialize lists to store the silhouette scores
    silhouette_scores = []

    # Iterate over different values of k
    for k in k_values:
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(transformed)

        # Get the cluster labels assigned to each data point
        cluster_labels = kmeans.labels_

        # Calculate the silhouette score
        silhouette = silhouette_score(transformed, cluster_labels)

        # Store the silhouette score
        silhouette_scores.append(silhouette)

    # Plot the silhouette scores for different values of k
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Cluster Numbers')
    plt.show()

def visualise_save_clustering(transformed, out_file = 'cluster_output_kmeans.pkl'):
    k = 7  # Number of clusters

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(transformed)

    # Access the cluster labels assigned to each data point
    cluster_labels = kmeans.labels_

    # Access the cluster centroids
    cluster_centers = kmeans.cluster_centers_

    x = transformed[:, 0]  # x-axis values
    y = transformed[:, 1]  # y-axis values

    # Create a scatter plot with different colors for each cluster
    scatter = plt.scatter(x, y, c=cluster_labels)
    centroids = np.array(cluster_centers)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', label='Centroids')

    legend_elements = scatter.legend_elements()
    unique_colors = legend_elements[0]
    plt.legend(unique_colors, [1,2,3,4,5,6,7])

    # Add labels and title
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Cluster Analysis Results')

    # Show the plot
    plt.show()

    with open(out_file, 'wb') as f:
        pickle.dump(kmeans, f)

'''save nested list of the cluster results 
 - quadruples per cluster
 - reduced matrix/plot points per cluster
 (uses more_than_one and transformed from main!)'''
def save_cluster_subgroups(cluster_labels):
    grouped_lists = [[] for _ in range(max(cluster_labels) + 1)]
    grouped_points = [[] for _ in range(max(cluster_labels) + 1)]

    # Iterate over the cluster_index and more_than_one lists using zip()
    for cluster_index, quad, trans in zip(cluster_labels, more_than_one, transformed):
        # Append the quad to the corresponding list based on the cluster_index
        grouped_lists[cluster_index].append(quad)
        grouped_points[cluster_index].append(trans)
    grouped_points = [np.array(group) for group in grouped_points]

    with open('cluster_output_words.pkl', 'wb') as f:
        pickle.dump(grouped_lists, f)

    with open('cluster_output_points.pkl', 'wb') as f:
        pickle.dump(grouped_points, f)

def print_clusters_to_file(word_clusters):
    for i, cluster in enumerate(word_clusters):
        with open(f'cluster_{i}.txt', 'w') as f:
            f.writelines(str(item) + '\n' for item in cluster)


if __name__ == '__main__':
    #save_stemmed_Counter() ...

    with open('stemmed_counter.pkl', 'rb') as f:
        stem_counter = pickle.load(f)

    # counter with stemmed quadruples appearing more than once
    with open('more_than_one_stemmed.pkl', 'rb') as f:
        transformed = pickle.load(f)
    # kmeans object
    with open('cluster_output_kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)

    # output from save_counter_subgroups, quadruples corresponding to points in plot
    with open('cluster_output_words.pkl', 'rb') as f:
        word_clusters = pickle.load(f)

    # output from save_counter_subgroups, points in plot
    with open('cluster_output_points.pkl', 'rb') as f:
        grouped_points = pickle.load(f)

    more_than_one = {key: value for key, value in stem_counter.items() if (value > 1)}

    matrix = create_distance_matrix(more_than_one)
    #save reduced dimension
    reduce_dimension(matrix)
    #calculate the eigenvalues
    eigenvalues(matrix)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    #grouped_points = np.array(grouped_points[0])

    #x = [arr[0] for arr in grouped_points[5]]  # visualize single clusters
    #y = [arr[1] for arr in grouped_points[5]]  # visualize single clusters
    x = [arr[0] for arr in transformed] #visualize full plot
    y = [arr[1] for arr in transformed] #visualize full plot
    language_keys = [x[0] for x in more_than_one] # 0= german, 1 = English, 2 = Spanish, 3 = Finnish

    top_words = [word for word, count in Counter(language_keys).most_common(11)]
    color_dict = {value: index + 1 for index, value in enumerate(top_words)}
    color_dict.update({'other': len(color_dict) + 1})
    color_indices = [color_dict[word] if word in color_dict else color_dict['other']for word in language_keys]
    #legend = [word if word in top_words else 'other' for word in language_keys]
    #print(color_indices)
    #color_indices = [list(set(german)).index(key) for key in german]

    # Create a scatter plot with different colors for each cluster
    scatter = plt.scatter(x, y, c = color_indices, cmap=cm.Set1)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    #legend_labels = [x for i, x in enumerate(legend) if x not in legend[:i]]
    #legend_labels.append('other')
    #print(legend_labels)
    plt.legend(handles=scatter.legend_elements()[0], labels=color_dict.keys(), title="German stems") #list(set(language_keys)) #list(set(top_words))
    plt.show()





