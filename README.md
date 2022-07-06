# Solar Panel Distribution Analysis
This repo documents the technical thinking behind the recommended points for the construction of the grid in the **[Perth](https://en.wikipedia.org/wiki/Perth)** area by solar panel distribution. 😎 This project mainly documents the basic source code and the data involved in the project will not be disclosed.
 
 <img width=1120 height=210 src="https://github.com/Robert-Mar/Solar-Panel-Distribution-Analysis/blob/main/images/solar_list.jpg">
 
## Steps of Analysis
This section mainly documents how to make recommendation for the construction of grid based on the distribution of custumer density in selected area. There are four main steps, including:
### Step 1: Define the targe range.
The grey boxes are selected for the area in the Perth, totalling 20 square kilometres. 

<img width=370 height=350 src="https://github.com/Robert-Mar/Solar-Panel-Distribution-Analysis/blob/main/images/scope.png">

### Step 2: Detect the solar panel
Crawl the remote sensing images from Google Map (20 layers) and store the images based on the latitude and longitue information. Then, we use the off-the-shelf object detection framework to detect the solar panel for each images. For more details, please refer to this [repo](https://github.com/Robert-Mar/Solar-Panel-Rotator). The actual latitude and longitude are inverted based on the relative position of the electric panel in the image. The final detected targets are displayed in the image below.

<img width=530 height=300 src="https://github.com/Robert-Mar/Solar-Panel-Distribution-Analysis/blob/main/images/scope_panel_v2.jpg">

### Step 3: Cluster the centers with K-Means
In order to find the potential grid construction address based on customer density and construction distance, we use [K-Means](https://en.wikipedia.org/wiki/K-means_clustering) calculate the feature centers of K clusters. K is determined by the number of grid that company plans to build. Specifically, we see each solar panel in the map as a point and then store them as a matrix, with their distribution remaining constant. As the image shown below, the points with different colours belong to different grid ranges.

<img width=500 height=400 src="https://github.com/Robert-Mar/Solar-Panel-Distribution-Analysis/blob/main/images/kmeans.png">

### Step 4: Calculate the potential grid address
After obtaining the cluster centers, we convert the relative position of the cluster center to the real position in the world coordinate system. We have marked potential grid construction sites using different icons, as shown below.

<img width=500 height=400 src="https://github.com/Robert-Mar/Solar-Panel-Distribution-Analysis/blob/main/images/grid_kmeans.png">
