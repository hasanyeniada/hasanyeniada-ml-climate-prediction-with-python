#Hasan YENİADA_220201024_Homework2
#Because of memory error occurance,I used first 1000 datas of dataset to apply dimensionality reduction techniques

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA

# Creating dataset
def createDataset(fileName):
    dataset = pd.read_csv(fileName, sep=r'\s{1,}',engine='python',usecols=["DIR", "SPD","CLG","SKC","L","M","H","TEMP","DEWP","SLP","STP","MAX","MIN","PCP06","PCPXX"])
    dataset = dataset.iloc[:, :].values

    for i in range(len(dataset)):
        for j in range(0,len(dataset[i])):
            if("T" in dataset[i][j] or "*" in dataset[i][j]):
                dataset[i][j] = "NaN" # I made these datas NaN to apply Imputer easily to take care of missing datas!

    return dataset[0:1000, :] #Because of memory error occurance,I used first 1000 datas

# Taking care of missing data
def takeCareOfMissingDatas(dataset):
    imputer = Imputer(missing_values = "NaN", strategy = 'most_frequent', axis = 0) #missing datas will be most frequent data in a column
    imputer = imputer.fit(dataset[:, 0:3])
    dataset[:, 0:3] = imputer.transform(dataset[:, 0:3])
    imputer = imputer.fit(dataset[:, 4:16])
    dataset[:, 4:16] = imputer.transform(dataset[:, 4:16])
    return dataset

# Encoding categorical data
def encodeCategoricalData(dataset):
    labelencoder_dataset = LabelEncoder()
    dataset[:, 3] = labelencoder_dataset.fit_transform(dataset[:, 3])
    onehotencoder = OneHotEncoder(categorical_features = [3])
    dataset = onehotencoder.fit_transform(dataset).toarray()

    # Feature Scaling
    sc = StandardScaler()
    dataset = sc.fit_transform(dataset)

    return dataset

# Applying PCA
def applyPCA(dataset, number_of_component):
    pca = PCA(number_of_component)
    dataset = pca.fit_transform(dataset)
    explained_variance = pca.explained_variance_ratio_
    print("PCA resulting explained variance matrix: ", explained_variance)

    total_explained_variance = 0
    for x in explained_variance:
        total_explained_variance += x
    print("After PCA, extracted ",  number_of_component,  " features preserves %", (total_explained_variance*100), " of data!")
    print("PCA has been completed!")
    visualizeResult(dataset, "PCA result")

# Applying MDS
def applyMDS(dataset, number_of_component):
    mds = MDS(number_of_component)
    dataset = mds.fit_transform(dataset)
    print("MDS has been completed!")
    visualizeResult(dataset, "MDS result")

# Applying Isomap
def applyIsomap(dataset,):
    iso = Isomap(n_neighbors=6, n_components=2)
    iso.fit(dataset)
    dataset = iso.transform(dataset)
    print("Isomap has been completed!")
    visualizeResult(dataset, "Isomap result")

# Applying LLE
def applyLLE(dataset):
    locallyLinearEmbedding = LocallyLinearEmbedding(n_components=2, n_neighbors=6, method='standard')
    dataset = locallyLinearEmbedding.fit_transform(dataset)
    print("LLE has been completed!")
    visualizeResult(dataset, "LLE result")

def visualizeResult(dataset, title, x_label="x1", y_label="x2", color ="r"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.scatter(dataset[:, 0], dataset[:, 1], s = 1, marker='.', c=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()

# Randomized PCA: Optional!
def applyRandomizedPCA(dataset):
    model = RandomizedPCA(100).fit(dataset)
    plt.plot(np.cumsum(model.explained_variance_ratio_))
    plt.xlabel('n components')
    plt.ylabel('cumulative variance')
    plt.show()

def main():
    dataset = createDataset("53727641925dat.txt")
    dataset = takeCareOfMissingDatas(dataset)
    dataset = encodeCategoricalData(dataset)
    print("Dataset has been created successfully and 1000 datas of dataset is taken to apply dimensinality techniques...")
    print("Dimensionality reduction operations are starting...")
    print("NOTE!!:Dimensionality reduction will be applied to 1000 datas because of memory error...")

    #applyRandomizedPCA(dataset)
    #applyPCA(dataset, 2)
    applyPCA(dataset, 10)
    #applyMDS(dataset, 2)
    #applyIsomap(dataset)
    #applyLLE(dataset)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Execution time:" + str(format((time.time() - start_time), '.2f')) + " seconds")


"""
        COMMENTS ON RESULT

First, comparison of execution times of techniques: (with 1000 datas!)       Note:I tried these exec. times without visualization,
PCA: 0.77 seconds                                                            becaue when plot is shown, timing continues while you are looking at plot!
MDS: 36.18 seconds
Isomap: 1.26 seconds
LLE: 0.94 seconds

Before applying these techniques, to have an idea about how many linear features are required to describe the data,
I applied randomized PCA and it shown me that nearly 10 components are required to preserve 92% of the variance.
(Note: I also added the plotting of that experiment and if you wish, you can see it with applyRandomizedPCA method!)
Then, I understood that if I use PCA and reduce dimensionality to 2, I cannot describe all data with 2 linear features.
At that point Isomap and LLE is required to find a small number of features that represent a large number of observed dimensions.
Because, with Isomap and LLE, we can reflect the geometric structure of the data points and so we can preserve more variance.
That result has also shown me that our data is not intrinsically very high dimensional and because of that reason 2 feature preserved
%39 of variance.At below, I added my comments on results after applying each technique:

Let's begin with PCA.After applying PCA, and choosing number of components = 2, resulting explained variance nearly %39
(as we can see that cumulative_variance graph.png), and we cannot describe all dataset with that 2 features and I increased
component number to 10 and result became %92. Because of we cannot visualize 11d data, I added PCA resulting plot with number of component = 2.
As we can see from graph we cannot explain the non-linear(geometric) structure of our data.If you wish, you can change number_of_component one by one and compare 
results with cumulative_variance graph which I also added to zip. The reason behind PCA does not perform so well is that there is a non-linear relationships 
between data points in dataset.

Then I applied MDS, as we know, MDS preserves Euclidean distances between points.We see from resulting MDS graph that nearby points remain nearby
and distant points remains distant. But, MDS also do not describe the geometric structure of our dataset. When I compare PCA and MDS graphs,
there is a bit differences, and I think main reason behind that is PCA preserves covariance of data, but MDS preserves euclidean distance and in our dataset,
covariance in our dataset is not equal euclidean distance between data points in high dimension so much, so graphs are a bit different.

Third, I applied Isomap which is an non-linear dimensionality reduction technique. We see from resulting Isomap grapgh that nearby points in high dimension
are not nearby anymore in low dimension.Isomap preserves geodestic distances between points and if points are near to each other according to Eucidean distance,
but they are distant according to geodestic distance, Isomap can reflect that intrinsic similarity between data points and reflect us the geometric structure of dataset.

LLE is local version of Isomap and approximates data by a set of linear patches.Neighborhood relationships between patches are preserved and this also seen from
resulting LLE graph.We see from graph that how LLE preserves the distances between neighboring points.

In these non-linear techniques, to determine optimal number of components is difficult, otherwise PCA lets you find the 
optimal dimension based on the explained variance.In our case that dimension is 11.

"""