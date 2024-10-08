# IT5100F - Assignment 3
Semester: AY2024/2025 Semester 1

- [IT5100F - Assignment 3](#it5100f---assignment-3)
- [**Task 1: Unsupervised Learning**](#task-1-unsupervised-learning)
  - [Import Libraries](#import-libraries)
  - [0: Data loading](#0-data-loading)
    - [Load the data from previous assignments](#load-the-data-from-previous-assignments)
  - [1: Data Preprocessing for Clustering](#1-data-preprocessing-for-clustering)
    - [1.1 Filter Data for Specific Sport](#11-filter-data-for-specific-sport)
    - [1.2 Generate Average Speed Dataset](#12-generate-average-speed-dataset)
    - [1.3 Compute Total Workout Time](#13-compute-total-workout-time)
    - [1.4 Compute Total Distance Covered](#14-compute-total-distance-covered)
    - [1.5 Merge Processed Data](#15-merge-processed-data)
  - [2: Determine the Optimal Number of Clusters](#2-determine-the-optimal-number-of-clusters)
    - [2.1 Run K-Means for Different Cluster Numbers](#21-run-k-means-for-different-cluster-numbers)
    - [2.2 Elbow Method for Optimal Clusters](#22-elbow-method-for-optimal-clusters)
  - [3: Cluster Analysis and Visualization](#3-cluster-analysis-and-visualization)
    - [3.1 Identify the Cluster Number for Each User](#31-identify-the-cluster-number-for-each-user)
    - [3.2 Visualize the Clusters](#32-visualize-the-clusters)
    - [3.3 Identify Similar Users](#33-identify-similar-users)
- [**Task 2: Free-form Exploration**: Sport Classfication](#task-2-free-form-exploration-sport-classfication)
- [Problem definition](#problem-definition)
- [Solution](#solution)
  - [Import Libraries](#import-libraries-1)
  - [0: Data loading](#0-data-loading-1)
    - [Load the data from previous assignments](#load-the-data-from-previous-assignments-1)
    - [1.1 Calculate Useful Factors](#11-calculate-useful-factors)
    - [1.2 Explore class distributions after aggregation](#12-explore-class-distributions-after-aggregation)
    - [1.3 Drop sports with very few users](#13-drop-sports-with-very-few-users)
  - [2: Model Training and Application](#2-model-training-and-application)
    - [2.1 Define the Features and Target Variable](#21-define-the-features-and-target-variable)
    - [2.2 Split the data into training and testing sets](#22-split-the-data-into-training-and-testing-sets)
    - [2.3 Build the Random Forest Classifier](#23-build-the-random-forest-classifier)
    - [2.4 Calculate the prediction accuracy](#24-calculate-the-prediction-accuracy)
    - [3.1 Try using SMOTE and retrain the model](#31-try-using-smote-and-retrain-the-model)

# **Task 1: Unsupervised Learning**

## Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
tqdm.pandas()
```



## 0: Data loading

### Load the data from previous assignments

For the project, we will continue to use the expanded dataset produced in Assignment 1. Here for the convenience of project development and evaluation, we have two options for data loading, one is loading the dataset from Google Drive(Option 1), another is loading dataset from our server(Option 2).


```python
option = 2 # 1 for Google Drive, 2 for url loading
```


```python
if option == 1:
  from google.colab import drive
  drive.mount('/content/drive')
  endomondo_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/IT5100F/endomondo_proper_cleaned_expanded.csv")
elif option == 2:
  url = "https://nextcloud.wu.engineer/index.php/s/md5dL6XXjjBYBRG/download/endomondo_proper_cleaned_expanded.csv"
  storage_options = {'User-Agent': 'Mozilla/5.0'}
  endomondo_df = pd.read_csv(url, storage_options=storage_options)
else:
  raise ValueError("Invalid option. Please choose 1 or 2.")
```

By displaying the first few rows of the dataset, we can observe the types of features it provides and gain a basic understanding of the dataset.


```python
endomondo_df.head()

```

The `info()` method prints information about the DataFrame, including the index dtype, columns, non-null values, and memory usage. It also provides a concise summary of the endomondo_df for us.


```python
endomondo_df.info()
```

## 1: Data Preprocessing for Clustering

### 1.1 Filter Data for Specific Sport

First, we need to filter the dataset to include only users participating in the sport _Bike_. Before applying the filter, we use the `value_counts()` method to identify the different types of sports and the number of entries for each sport in the dataset.


```python
sports_counts = endomondo_df['sport'].value_counts()
print(sports_counts)

```

Here, the endomondo_df dataset includes various sports such as `bike`, `run`, and `bike (transport)`. To filter out the entries where the sport is `bike`, we use `==` to generate a boolean sequence. This sequence is then used to index the dataframe, returning only the rows where the sport is `bike`. We apply the `copy()` method to create an independent copy of the original dataframe, ensuring any modifications made later won't affect the original data.


```python
# Filter the dataset to include only users participating in the sport "bike"
bike_df = endomondo_df[endomondo_df['sport'] == 'bike'].copy()
```

Before proceeding to the next step, we notice that the data types of the `id` and `timestamp` fields are `float` and `str`, respectively, which are not appropriate. We will convert them to `int` and `datetime` types.


```python
print(type(bike_df.loc[0,'id']))
print(type(bike_df.loc[0,'timestamp']))
```


```python
# Convert 'timestamp' column to datetime objects
bike_df['timestamp'] = pd.to_datetime(bike_df['timestamp'])

# Convert 'id' colum to int
bike_df['id'] = bike_df['id'].astype(int)
```

Let's check the type again.


```python
bike_df.info()
```


```python
bike_df.head()
```

Although we dropped all the NA values in Assignment 1, we will check for NA values again to ensure no NA values are present in the data.


```python
bike_df.isna().sum().sum()
```

At this point, we have successfully obtained a subset of the original dataset that only includes users who participated in sports `bike`.

### 1.2 Generate Average Speed Dataset

Create a new dataset with `user_ids` and their average speed (`avg_speed`).

Here we notice the corresponding field of `user_ids` is `id`. Using the `groupby()` method, we group all entries by `id`. Then, we select the `speed` column to return only speed values, and apply the `mean()` method to calculate the average speed. The result is a DataFrame where `id` is the index and the average speed is the value. Finally, we use `reset_index()` to make `id` a separate column and rename the speed column to `avg_speed`.


```python
# Create a new dataset with user_ids and their average speed (avg_speed)

avg_speed_df = bike_df.groupby('id')['speed'].mean().reset_index()
avg_speed_df = avg_speed_df.rename(columns={'speed': 'avg_speed'})
print(len(avg_speed_df))
avg_speed_df.head()

```

### 1.3 Compute Total Workout Time

Create a new dataset with `user_ids` and their total workout time (`workout_time`), calculated as the difference between the minimum and maximum timestamp in seconds for each user.

First, we group the DataFrame by `id` and aggregate the timestamp values using the `agg()` function to compute both the minimum and maximum timestamps for each user. We use `reset_index()` to flatten the grouped data into a DataFrame with a new index.

Next, we calculate the workout duration by finding the difference between the maximum and minimum timestamps for each user and convert this difference into seconds using `dt.total_seconds()`. We store this result in a new column called `workout_time`.

Finally, we retain only the `id` and `workout_time` columns showing the total workout time for each user.


```python
# Create a new dataset with user_ids and their total workout time (workout_time),
# calculated as the difference between the minimum and maximum timestamp in seconds for each user.


# Group by user ID and calculate the total workout time
workout_time_df = bike_df.groupby('id')['timestamp'].agg(['min', 'max']).reset_index()
workout_time_df['workout_time'] = (workout_time_df['max'] - workout_time_df['min']).dt.total_seconds()
workout_time_df = workout_time_df[['id', 'workout_time']]

workout_time_df.head()

```

### 1.4 Compute Total Distance Covered

Create a new dataset with user_ids and the total distance (total_distance) covered by each user.

To obtain total distance for each user, we use $\text{distance}=\text{avg_speed} \times \text{workout_time}$ formula. So first, we merge the previous two DataFrame, then multiply `avg_speed` column and `workout_time` column to get `total_distance`.


```python
# Create a new dataset with user_ids and the total distance (total_distance) covered by each user.
total_distance_df = pd.merge(workout_time_df, avg_speed_df, on='id')
total_distance_df['total_distance'] = total_distance_df['avg_speed'] * total_distance_df['workout_time']
total_distance_df = total_distance_df[['id', 'total_distance']]
total_distance_df.head()
```

### 1.5 Merge Processed Data

Merge all the above datasets into a single data frame called `user_merged_df`.
To merge all the DataFrame together, We set `id` as index, then use `pd.concat` function to concatenate three DataFrame together.


```python
# Merge all the above datasets into a single data frame called user_merged_df.

avg_speed_df.set_index("id", inplace=True)
total_distance_df.set_index("id", inplace=True)
workout_time_df.set_index("id", inplace=True)

user_merged_df = pd.concat(
    [avg_speed_df, workout_time_df, total_distance_df], axis=1
).reset_index()

user_merged_df.head()
```

## 2: Determine the Optimal Number of Clusters

### 2.1 Run K-Means for Different Cluster Numbers

Use the user_merged_df to run k-means clustering from 2-11 clusters.

To perform k-means clustering, the `id` field should not be used as a feature, so we drop it first. Then, we use `StandardScaler` to remove the mean and scale the data to unit variance. Since k-means uses Euclidean distance, features with larger ranges would dominate the distance calculation if we don't apply standard scaling.

Next, we run clustering with 2 to 11 clusters and collect the inertia values. We set `random_state` to ensure reproducible results.


```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = user_merged_df.drop(columns=["id"])
scaler = StandardScaler()
X = scaler.fit_transform(X)

inertia = []
cluster_range = range(2, 12)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

print(inertia)
```

### 2.2 Elbow Method for Optimal Clusters

Plot the inertia (y-axis) against the number of clusters (x-axis) to identify the optimal number of clusters using the Elbow method.

The Elbow Method helps in selecting the number of clusters by identifying where the inertia starts to decrease more slowly.

From our graph, inertia starts to decrease more slowly when the cluster number comes to 6. So we think the best cluster number is 6.


```python
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o', linestyle='-', color='b')
plt.title('K-Means Clustering Inertia for 2 to 11 Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(cluster_range)
plt.grid(True)
plt.show()
```


```python
best_cluster_number = 6
```

## 3: Cluster Analysis and Visualization

### 3.1 Identify the Cluster Number for Each User

Create a k-means model using the features from `user_merged_df` and the optimal number of clusters identified in Task 2.

Add a new column to the data frame called `cluster` that contains the cluster number for each user.


```python
# Create a k-means model using the features from user_merged_df and the optimal number of clusters identified in Task 2.

kmeans = KMeans(n_clusters=best_cluster_number, random_state=42)
kmeans.fit(X)


# Add a new column to the data frame called cluster that contains the cluster number for each user.
user_merged_df['cluster'] = kmeans.labels_
user_merged_df.head()

```

### 3.2 Visualize the Clusters

Visualize the clusters using a scatter plot.

Because there are three features in the dataset, we choose 3D scatter plot to display the result.


```python
# Visualize the clusters using a scatter plot. (3d)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(user_merged_df['avg_speed'],
                     user_merged_df['total_distance'],
                     user_merged_df['workout_time'],
                     c=user_merged_df['cluster'],
                     cmap='viridis')

ax.set_xlabel('Average Speed')
ax.set_ylabel('Total Distance')
ax.set_zlabel('Workout Time')
ax.set_title('3D Visualization of Clusters')

plt.colorbar(scatter)
plt.show()
```

For better visualization, we first apply Principal Component Analysis to reduce the dimensionality of the features from 3 to 2. This allows us to project the data into two principal components, which can then be visualized using a 2D scatter plot.


```python
# Use PCA to reduce dimension from 3 to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['cluster'] = user_merged_df['cluster'].values

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', palette='viridis', data=pca_df)
plt.title('PCA - 2D Visualization with Cluster Colors', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.grid(True)
plt.show()
```

Then we try t-SNE method for dimensionality reduction. In our practice, we reduce the features to 2 components for visualization using a 2D scatter plot.


```python
# Use t-SNE for dimension reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(X)
user_merged_df['tsne1'] = tsne_result[:, 0]
user_merged_df['tsne2'] = tsne_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='tsne1', y='tsne2', hue='cluster', data=user_merged_df, palette='viridis')
plt.title('t-SNE of Clusters')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
```

### 3.3 Identify Similar Users

Determine which cluster the user ID: 377398220 belongs to.

Identify similar users (workout buddies) within that cluster so that you can make recommendations based on the shared workout patterns.

To find which cluster a user belongs to, we use the == operation on the id field.

After identifying the user's cluster, we retrieve users from the same cluster (except the target user with ID 377398220). Next, we compute the similarity score using the Euclidean distance between the target user and each user in the same cluster. We calculate the Euclidean distance based on relevant features (`avg_speed` and `total_distance`) and then sort users by this distance. This allows us to identify the top-K users who are most similar to the target user based on the Euclidean distance.




```python
target_user_id = 377398220
target_user = user_merged_df[user_merged_df['id'] == target_user_id].iloc[0]
target_user_cluster = target_user['cluster']

similar_users = user_merged_df[
    (user_merged_df['cluster'] == target_user_cluster) &
    (user_merged_df['id'] != target_user_id)
].copy()

target_user_features = target_user[['avg_speed', 'total_distance']].values

similar_users['similarity'] = similar_users.apply(
    lambda row: euclidean(row[['avg_speed', 'total_distance']], target_user_features),
    axis=1
)

similar_users = similar_users.sort_values('similarity')

print(f"Target User (ID: {target_user_id}):")
print(f"Cluster: {target_user_cluster}")
print(f"Average Speed: {target_user['avg_speed']:.2f} km/h")
print(f"Total Distance: {target_user['total_distance']:.2f} km")
print(f"Workout Time: {target_user['workout_time']:.2f} seconds")
print("\nTop 5 similar users:")
for _, user in similar_users.head().iterrows():
    print(f"User ID: {user['id']}")
    print(f"  Average Speed: {user['avg_speed']:.2f} km/h")
    print(f"  Total Distance: {user['total_distance']:.2f} km")
    print(f"  Workout Time: {user['workout_time']:.2f} seconds")
    print(f"  Similarity Score: {user['similarity']:.2f}")
    print()

cluster_avg = user_merged_df[user_merged_df['cluster'] == target_user_cluster].mean()
print(f"Cluster {target_user_cluster} Averages:")
print(f"Average Speed: {cluster_avg['avg_speed']:.2f} km/h")
print(f"Total Distance: {cluster_avg['total_distance']:.2f} km")
print(f"Workout Time: {cluster_avg['workout_time']:.2f} seconds")
```

# **Task 2: Free-form Exploration**: Sport Classfication


# Problem definition


The objective of this task is to develop a machine learning model capable of classifying different types of sports activities based on aggregated user data from the Endomondo fitness tracking application. This classification problem presents several interesting challenges and considerations:

1. **Dataset**:
   - We will be using the `endomondo_proper_cleaned_expanded.csv` dataset, which contains detailed tracking information for various sports activities.

2. **Feature Selection**:
   Our classification will be based on the following key factors:
   - Average altitude of each user
   - Average heart rate of each user
   - Average latitude and longitude of each user
   - Average speed of each user
   - Total time spent doing sports for each user

3. **Data Preprocessing**:
   - We will aggregate the data by calculating the average of each factor per user. This approach serves two purposes:
     * It reduces the size of the dataset, making it more manageable for analysis.
     * It provides a holistic view of each user's activity patterns, which may be more indicative of the type of sport they engage in most frequently.

4. **Class Imbalance**:
   - The dataset likely exhibits imbalance among different sports classes, with some activities being more common than others. This imbalance needs to be addressed to ensure fair classification across all sports.

5. **Geospatial Considerations**:
   - The inclusion of average latitude and longitude introduces a geospatial element to our classification. We need to consider how geographical location might influence sport classification.

6. **Temporal Aspects**:
   - The total time spent on sports activities provides a temporal dimension to our data. We should consider how this might interact with other features (e.g., average speed) to distinguish between different sports.

7. **Feature Scaling**:
   - Given the diverse nature of our features (altitude, heart rate, geographical coordinates, speed, and time), appropriate scaling or normalization will be crucial to ensure all features contribute fairly to the classification.

8. **Model Selection and Evaluation**:
   - We need to select an appropriate machine learning algorithm that can handle multi-class classification with potentially imbalanced data.
   - Evaluation metrics should go beyond simple accuracy, considering precision, recall, and F1-score for each sport category.

This model could have practical applications in personalizing fitness recommendations, improving activity recognition in tracking apps, and gaining insights into how different sports are characterized by various physiological and environmental factors.

# Solution

## Import Libraries


```python
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
```

## 0: Data loading

### Load the data from previous assignments

For the project, we will continue to use the expanded dataset produced in Assignment 1. Here for the convenience of project development and evaluation, we have two options for data loading, one is loading the dataset from Google Drive(Option 1), another is loading dataset from our server(Option 2).


```python
option = 2 # 1 for Google Drive, 2 for url loading
```


```python
if option == 1:
  from google.colab import drive
  drive.mount('/content/drive')
  endomondo_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/IT5100F/endomondo_proper_cleaned_expanded.csv")
elif option == 2:
  url = "https://nextcloud.wu.engineer/index.php/s/md5dL6XXjjBYBRG/download/endomondo_proper_cleaned_expanded.csv"
  storage_options = {'User-Agent': 'Mozilla/5.0'}
  endomondo_df = pd.read_csv(url, storage_options=storage_options)
else:
  raise ValueError("Invalid option. Please choose 1 or 2.")
```


```python
endomondo_df.head()

```


```python
endomondo_df.info()
```

### 1.1 Calculate Useful Factors



```python
# Convert 'id' column to int
endomondo_df['id'] = endomondo_df['id'].astype("Int64")

# Calculate the average value of useful columns
avg_df = endomondo_df.groupby(['id','sport'])[['altitude', 'heart_rate', 'latitude', 'longitude', 'speed']].mean()
avg_df = avg_df.rename(columns={'altitude': 'avg_altitude','heart_rate': 'avg_heart_rate','latitude': 'avg_latitude','longitude': 'avg_longitude','speed': 'avg_speed'})

# Convert 'timestamp' column to datetime objects
endomondo_df['timestamp'] = pd.to_datetime(endomondo_df['timestamp'])

# Calculate the total time duration
total_time_df = endomondo_df.groupby(['id','sport'])['timestamp'].agg(['min', 'max'])
total_time_df['total_time'] = (total_time_df['max'] - total_time_df['min']).dt.total_seconds()

# Merge the two dataframes
new_endomondo_df = pd.merge(avg_df, total_time_df, on=['id', 'sport']).reset_index()

# Drop the useless 'min' and 'max' columns
new_endomondo_df = new_endomondo_df.drop(columns=['min', 'max'])

new_endomondo_df.head()
```

### 1.2 Explore class distributions after aggregation


```python
class_distribution = new_endomondo_df['sport'].value_counts()
print("Class Distribution:")
print(class_distribution)
plt.figure(figsize=(12, 6))
class_distribution.plot(kind='bar')
plt.title('Distribution of Sports Classes')
plt.xlabel('Sport')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

From the distribution above, we can observe that many sport classes have very few data samples of unique users after aggregation. These data cannot provide enough information for classfication, instead, these may introduce noise to our dataset. Hence, we decide to remove those sports with samples less than 100.

### 1.3 Drop sports with very few users


```python
# Count the number of samples for each sport
sport_counts = new_endomondo_df['sport'].value_counts()

# Keep only the sports with at least 100 user samples
sports_to_keep = sport_counts[sport_counts >= 100].index

# Filter the dataframe to keep only the selected sports
filtered_df = new_endomondo_df[new_endomondo_df['sport'].isin(sports_to_keep)]
```


```python
class_distribution = filtered_df['sport'].value_counts()
print("Class Distribution:")
print(class_distribution)
plt.figure(figsize=(12, 6))
class_distribution.plot(kind='bar')
plt.title('Distribution of Sports Classes')
plt.xlabel('Sport')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

After filtering, we are actually classifying five sports:

1.  bike
2.  run
3.  bike(transport)
4. mountain bike
5. indoor cycling.

using the aggregation features we got from each user.


## 2: Model Training and Application

### 2.1 Define the Features and Target Variable


```python
# Features to use as input for the model
X_Features = ['avg_altitude', 'avg_heart_rate','avg_latitude','avg_longitude','avg_speed','total_time']

# Features to predict
Y_Features = ['sport']

X = filtered_df[X_Features]
Y = filtered_df[Y_Features]
```

### 2.2 Split the data into training and testing sets


```python
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Split the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2.3 Build the Random Forest Classifier


```python
rf_N=100

# Initialization Random Forest Classifier
rf = RandomForestClassifier(n_estimators=rf_N, random_state=RANDOM_STATE)

# Training Model
rf.fit(X_train_scaled, Y_train)
```

### 2.4 Calculate the prediction accuracy


```python
# Prediction
Y_pred = rf.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(Y_test, Y_pred))

# Calculate Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

# Normalize Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, None]

# Draw Normalized Confusion Matrix through seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted Sports')
plt.ylabel('Actual Sports')
plt.title('Normalized Confusion Matrix of Sports')
plt.show()
```

Based on the classification report, the model performs well overall with an accuracy of 0.95 and a weighted average F1-score of 0.95. However, the macro average F1-score (0.80) is lower than the weighted average (0.95), indicating that the model's performance varies significantly across classes, with poorer performance on minority classes. Performance for "bike (transport)", "indoor cycling", and "mountain bike" is notably lower, with F1-scores ranging from 0.54 to 0.76.

To address the class imbalance issue and potentially improve the model's performance on minority classes, we plan to use SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority classes. This approach will create synthetic examples of the underrepresented classes, balancing the dataset.

By applying SMOTE and retraining the model, we aim to:

* Improve the F1-scores for the minority classes (bike transport, indoor cycling, mountain bike).
* Achieve a more balanced performance across all classes, potentially increasing the macro average F1-score.
* Maintain or possibly improve the overall accuracy while ensuring better representation of all sports categories.

### 3.1 Try using SMOTE and retrain the model


```python
# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, Y_train)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions and evaluate the model
Y_pred = rf_classifier.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(Y_test, Y_pred))

# Calculate Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

# Normalize Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, None]

# Draw Normalized Confusion Matrix through seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted Sports')
plt.ylabel('Actual Sports')
plt.title('Normalized Confusion Matrix of Sports')
plt.show()
```

The use of SMOTE has indeed improved the classification results, particularly for the minority classes. Let's break down the improvements and discuss real-life applications:

Improvements after SMOTE:

1. Minority classes performance:
   - Bike (transport): F1-score improved from 0.73 to 0.79
   - Indoor cycling: F1-score slightly decreased from 0.76 to 0.72, but recall improved from 0.68 to 0.75
   - Mountain bike: F1-score significantly improved from 0.54 to 0.65

2. Balanced performance:
   - Macro avg F1-score increased from 0.80 to 0.82, indicating better overall performance across all classes
   - Recall for minority classes also improved, showing better detection of these activities

3. Maintained overall accuracy:
   - The overall accuracy remained at 0.95, while improving minority class performance

Real-life applications of this classfication model:

1. Fitness tracking apps: The model can automatically classify different cycling activities, providing users with more accurate activity logs and tailored feedback.

2. Urban planning: By accurately distinguishing between regular cycling and transport cycling, city planners can better understand commuting patterns and design appropriate infrastructure.



This improved model shows the value of addressing class imbalance in real-world datasets, leading to more reliable and equitable performance across different activities.


```python

```
