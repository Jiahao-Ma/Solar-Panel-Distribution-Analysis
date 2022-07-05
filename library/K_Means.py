import numpy as np
import random
import matplotlib.pyplot as plt

def Marko_KMeans(K, X, IterSteps):
    """ KMeans method:
            Steps 1: Randomly choose centroids whose number depends on K
            Steps 2: Calculate the distance between centroids and rest of samples
            Steps 3: Judge the distance and classify the samples via the nearest distance between itself and centroids
            Steps 4: Repeat previous steps until they converge
            
        Params:
            K: The number of cluster
            X: Input features 
            IterSteps: The number of training steps
    """
    def _Init_Centroids(K, X):
        """ Randomly Select the unique centroids
            C: the unique centroids of input features
        """
        C = np.zeros((K, X.shape[1]))
        xShuffle = X.copy()
        xShuffle = np.unique(xShuffle)
        idxShuffle = np.arange(len(xShuffle))
        random.shuffle(idxShuffle)
        for i in range(K):
            C[i] = xShuffle[idxShuffle[i]] 
        return C 
    
    def _E_Steps(K, X, C):
        """ Calculate the L [the labels (centroid values) assigned to each sample in the dataset]
            Params:
                K: number of cluster
                C: clusters, eg: shape: (2, 4) number of cluster: 2, number of features: 4
                X: samples, eg: shape: (1000, 4) number of samples: 1000, number of features: 4
            Return:
                C: cluster after K-Means
                L: each row contain the information of cluster, eg: shape: (1000, 4) row contain the features of cluster, 
                    so they will have lots of overlaps
                minIndxList(list): list element store classification of cluster
        """
        L = np.zeros(X.shape)
        numCluster = K
        # (1000, 4) -> (1000, 8)
        xTile = np.tile(A = X, reps = numCluster)
        # (2, 4) -> (1, 8) -> (1000, 8)
        centerTile = np.tile(A = C.reshape(1,-1), reps = (len(X), 1))
        # (1000, 8) -> (1000, 2, 4)
        totalDiff = (xTile - centerTile).reshape(X.shape[0], numCluster, X.shape[1])
        # (1000, 2)
        distance = np.linalg.norm(totalDiff, axis=2)
        # (1000, 2) -> (1000, )
        minIdxList = np.argmin(distance, axis=1)
        # (1000, 4) = (2, 4)[(1000,)]
        L = C[minIdxList]
        return L, minIdxList
    
    def _M_step(X, C, minIndxList):
        """ Update the clusters (C)
            Params:
                C: clusters, eg: shape: (2, 4) number of cluster: 2, number of features: 4
                X: samples, eg: shape: (1000, 4) number of samples: 1000, number of features: 4
            Return:
                C: new clusters 
        """
        for idx in np.unique(minIndxList):
            points = X[idx == minIndxList]
            C[idx] = np.mean(points, axis=0)
        return C

    #############
    #  K-Means  #
    #############
    C = _Init_Centroids(K, X)
    for i in range(IterSteps):
        L, minIdxList = _E_Steps(K, X, C)
        C = _M_step(X, C, minIdxList)
    L, minIndxList = _E_Steps(K, X, C)
    return C, L, minIndxList

def normalize(X):
    """ Max-Min-Normalization:
        normalize the input samples whose input size: [N, C] 
        N: number of samples
        C: number of features of each sample

        Params:
            X: Input samples, `size`:[N, C]
        Return:
            X: Output samples with normalization, `size`:[N, C]
                features range from 0 to 1
    """
    numFeatures = X.shape[1]
    for i in range(numFeatures):
        maxFeature = X[:, i].max()
        minFeature = X[:, i].min()
        X[:, i] = (X[:, i] - minFeature) / (maxFeature - minFeature)
    return X

if __name__ == "__main__":

    def read_analysis_txt(fname, printInfo=True):
        panel_num = 0
        pool_num = 0
        panel_list = []
        pool_list = []
        with open(fname, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.split(',')
                if len(line) == 1:
                    continue
                else:
                    fname, bboxes = line[0], line[1:]
                    for box_indx in range(0, len(bboxes), 4):
                        label = bboxes[box_indx]
                        if label == 'panel':
                            panel_num += 1
                            panel_list.append(bboxes[box_indx: box_indx+4])
                        elif label == 'pool':
                            pool_num += 1
                            pool_list.append(bboxes[box_indx: box_indx+4])
        
        if printInfo:
            print('panel num: %d' %panel_num)
            print('pool num: %d' %pool_num)
            print('total target: %d' %(panel_num + pool_num))

        return panel_list, pool_list

    def get_relative_point(target, point, ratio=100000):
        lat = (target[0] - point[0]) * ratio
        lon = (point[1] - target[1]) * ratio
        return [lat, lon]

    analysis_path = r'E:\Solar\company_part1\predict_latitude_longitude\analysis.txt'
    scope = [[-32.208, 115.866], [-32.068, 115.866], [-32.068, 116.033], [-32.208, 116.033]]

    panel_list, pool_list = read_analysis_txt(analysis_path)

    relative_panel_list = []
    max_lat = 0
    max_lon = 0
    for panel in panel_list:
        label, lat, lon, prob = panel
        [lat, lon] = get_relative_point(scope[1], [float(lat), float(lon)])
        if lat > max_lat:
            max_lat = lat
        if lon > max_lon:
            max_lon = lon
        relative_panel_list.append([label, lat, lon, float(prob)])
    print('relative Image size:', int(max_lat), int(max_lon))

    w, h = 14000, 8000
    radius = 15
    scale = 80
    MapArray = np.zeros(shape=[14000, 8000, 3], dtype=np.uint8)
    relative_loc_list = []
    for panel in relative_panel_list:
        lat, lon = int(panel[1]), int(panel[2])
        relative_loc_list.append([lat, lon])
        MapArray[lat-radius:lat+radius, lon-radius: lon+radius] += scale
    
    MapArray = np.clip(MapArray, 0, 255)

    plt.figure(figsize=(15, 20))
    plt.subplot(121)
    plt.imshow(MapArray)

    IterSteps = 100
    K = 25
    relative_loc_list = np.array(relative_loc_list)
    
    X = relative_loc_list
    
    C, L, minIndxList = Marko_KMeans(K, X, IterSteps)
 
    random_color = (np.random.random((K, 3)) * 0.6 + 0.4).tolist()
    KMeansMapArray = np.zeros(shape=[14000, 8000, 3])
    
    for loc, cls in zip(relative_loc_list, minIndxList):
        lat, lon = loc
        KMeansMapArray[lat-radius:lat+radius, lon-radius: lon+radius, :] = np.array(random_color[cls]) * 255
    # KMeansMapArray = np.array(KMeansMapArray * 255)
    KMeansMapArray = np.clip(KMeansMapArray, 0, 255).astype(np.uint)
    plt.subplot(122)
    plt.imshow(KMeansMapArray)
    plt.show()
