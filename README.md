## CSCI_4120_HW2

Name: Naomi Jainarine Email: jainarinen20@students.ecu.edu

Name: Pusp Raj Bhatt Email: bhattp20@students.ecu.edu

# Quick Start:
Python 3.8.10 was used to run this code

In the first block of code, the following packages should be installed: 

```python
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas as pd

from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
```
We used a randomly generated data set for this classification project. This data set can be generated and visualized by using the following code:

```python
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50);
```
Based on this scatter plot, we can assume that the best k value is 4. The data points seem to naturally cluster in 4 groups. The centroids of these groups can be calculated and visualized by running the code below.

```python
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# predicted centroids
centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

kmeans.cluster_centers_.shape
```

At this point,  we have chosen a K value based on a visual representation. In order to mathematically calulate the best K value for our data set we can use KElbowVisualizer to determine the distortion score.

```python
model= KMeans()
visualizer = KElbowVisualizer(model,k=(2,12))

visualizer.fit(X)
visualizer.show()
```
The generated figure confirms that 4 is the best k. 

Next, we can calculate the accuracy of our K-means model. But first, you want to prepare your predictions by using the code below.

```python
#        create a zero matrix with the same shape
labels = np.zeros_like(clusters)

for i in range(4):
   
    mask = (clusters == i)
   
    labels[mask] = mode(y_true[mask])[0]
 ```

Now you can calculate the accuracy score using 'labels'

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, labels)
print(accuracy)
```
The accuracy for this KMeans model was 1. This means the predicted values all matched the observed values. This can be visualized with the sklearn.metrics package. The confusion matrix array was made into a data.frame before using the heat map visualization tool.

```python
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_true, labels)
print(mat)

mat_cm = pd.DataFrame(mat,range(4),range(4))
sns.heatmap(mat_cm, annot=True)


plt.xlabel('true label')
plt.ylabel('predicted label');

plt.show()
```
![Confussion Matrix](
