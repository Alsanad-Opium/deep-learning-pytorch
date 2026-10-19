let me just give you an overview to solidify my learning 

DBSCAN is a density based clustering algorithm that works well with noise and arbitary data. It has two important parameters that is epsilon and min_samples 

Epsilon (eps) is the maximum radius of the search neighborhood around any single point
• If a point finds min_samples neighbors within its own radius, it becomes a core point.
• Two core points end up in the same cluster if they are close enough to touch neighborhoods (i.e., they are within eps distance of each other)
min_samples is the required number of points that is required to form a cluster we have to pass this, we can calculate it using a formula that is 2*n ie number of columns 


To calculate the epsilon we can use the k-distance elbow graph with be use find the optimal number of distance to consider a point in a cluster 

we do this by using the nearestNeughbors class from sklearn 

we load the model which can perform very fast distance calculation and after creating the obj we can call the .fit() method that creates a tree that can acess the nearest neighbor instantly of a point

we then use the kneighbors on the map/tree to that iterates through each and every point and gives out the distance with the point and its index 

we can then use the distances last columns sort it in ascending order and plot on the graph with the y-axis containing distances and the x-axis the datapoints 
TO indentify the eps value you need to look in the graph where a elbow is being formed that is the value for eps 

also as it based on distance calculation make sure to scale the values accordingly