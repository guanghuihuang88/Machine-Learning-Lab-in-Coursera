# ex7  K-均值聚类算法和PCA主成分分析 K-Means Clustering and Principal Component Analysis

[toc]

### 1.  K-均值聚类算法 K-Means Clustering



#### 1.1  实现K-均值算法 Implementing K-means



- 为每一个训练样本找到邻近中心点

- 打开 findClosestCentroids.m 函数，做如下更改

  ```matlab
  function idx = findClosestCentroids(X, centroids)
  %FINDCLOSESTCENTROIDS computes the centroid memberships for every example
  %   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
  %   in idx for a dataset X where each row is a single example. idx = m x 1 
  %   vector of centroid assignments (i.e. each entry in range [1..K])
  %
  
  % Set K
  K = size(centroids, 1);
  
  % You need to return the following variables correctly.
  idx = zeros(size(X,1), 1);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Go over every example, find its closest centroid, and store
  %               the index inside idx at the appropriate location.
  %               Concretely, idx(i) should contain the index of the centroid
  %               closest to example i. Hence, it should be a value in the 
  %               range 1..K
  %
  % Note: You can use a for-loop over the examples to compute this.
  %
  
  for i = 1:size(idx)
      s = sum((X(i, :) - centroids) .^ 2, 2);
      minK = min(s);
      idx(i) = find(s == minK);
  end
  
  % =============================================================
  
  end
  ```
  
- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | K-Means Clustering and PCA...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==            Find Closest Centroids (k-Means) |  30 /  30 | Nice work!
  ==            Compute Centroid Means (k-Means) |   0 /  30 | 
  ==                                         PCA |   0 /  20 | 
  ==                          Project Data (PCA) |   0 /  10 | 
  ==                          Recover Data (PCA) |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  30 / 100 | 
  == 
  ```
  
- 上传数据并检验函数正确性

  ```matlab
  % Load an example dataset that we will be using
  load('ex7data2.mat');
  
  % Select an initial set of centroids
  K = 3; % 3 Centroids
  initial_centroids = [3 3; 6 2; 8 5];
  
  % Find the closest centroids for the examples using the initial_centroids
  idx = findClosestCentroids(X, initial_centroids);
  fprintf('Closest centroids for the first 3 examples: %d %d %d', idx(1:3))
  ```




- 更新每一个中心点为相应聚集样本的均值

- 打开 computeCentroids.m 函数，做如下更改

  ```matlab
  function centroids = computeCentroids(X, idx, K)
  %COMPUTECENTROIDS returns the new centroids by computing the means of the 
  %data points assigned to each centroid.
  %   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
  %   computing the means of the data points assigned to each centroid. It is
  %   given a dataset X where each row is a single data point, a vector
  %   idx of centroid assignments (i.e. each entry in range [1..K]) for each
  %   example, and K, the number of centroids. You should return a matrix
  %   centroids, where each row of centroids is the mean of the data points
  %   assigned to it.
  %
  
  % Useful variables
  [m n] = size(X);
  
  % You need to return the following variables correctly.
  centroids = zeros(K, n);
  
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Go over every centroid and compute mean of all points that
  %               belong to it. Concretely, the row vector centroids(i, :)
  %               should contain the mean of the data points assigned to
  %               centroid i.
  %
  % Note: You can use a for-loop over the centroids to compute this.
  %
  
  for k = 1:K
      centroids(k, :) = mean(X(find(idx == k), :), 1);
  end
  
  % =============================================================
  
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | K-Means Clustering and PCA...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==            Find Closest Centroids (k-Means) |  30 /  30 | Nice work!
  ==            Compute Centroid Means (k-Means) |  30 /  30 | Nice work!
  ==                                         PCA |   0 /  20 | 
  ==                          Project Data (PCA) |   0 /  10 | 
  ==                          Recover Data (PCA) |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  60 / 100 | 
  == 
  ```




#### 1.2  利用K-均值算法分类样本数据集 K-means on example dataset



- 在命令行输入

```matlab
% Load an example dataset
load('ex7data2.mat');
% Settings for running K-Means
max_iters = 10;

initial_centroids = [3 3; 6 2; 8 5];

% Run K-Means algorithm. The 'true' at the end tells our function to plot the progress of K-Means
figure('visible','on'); hold on; 
plotProgresskMeans(X, initial_centroids, initial_centroids, idx, K, 1); 
xlabel('Press ENTER in command window to advance','FontWeight','bold','FontSize',14)
[~, ~] = runkMeans(X, initial_centroids, max_iters, true);
set(gcf,'visible','off'); hold off;
```

![image-20200721141040428](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex7/image-20200721141040428.png)





### 2.  PCA主成分分析 Principal Component Analysis



#### 2.1  样本数据集 Example dataset



- 可视化样本数据集

  ```matlab
  % Initialization
  clear;
  % The following command loads the dataset. You should now have the variable X in your environment
  load ('ex7data1.mat');
  
  % Visualize the example dataset
  figure;
  plot(X(:, 1), X(:, 2), 'bo');
  axis([0.5 6.5 2 8]); axis square;
  ```
  
  ![image-20200723115420950](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex7/image-20200723115420950.png)
  
  

#### 2.2  实现PCA算法 Implementing PCA



- 打开 pca.m 函数，做如下更改

  ```matlab
  function [U, S] = pca(X)
  %PCA Run principal component analysis on the dataset X
  %   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
  %   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
  %
  
  % Useful values
  [m, n] = size(X);
  
  % You need to return the following variables correctly.
  U = zeros(n);
  S = zeros(n);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: You should first compute the covariance matrix. Then, you
  %               should use the "svd" function to compute the eigenvectors
  %               and eigenvalues of the covariance matrix. 
  %
  % Note: When computing the covariance matrix, remember to divide by m (the
  %       number of examples).
  %
  
  sigma = (1/m) * X' * X;
  [U, S, V] = svd(sigma);
  
  % =========================================================================
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | K-Means Clustering and PCA...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==            Find Closest Centroids (k-Means) |  30 /  30 | Nice work!
  ==            Compute Centroid Means (k-Means) |  30 /  30 | Nice work!
  ==                                         PCA |  20 /  20 | Nice work!
  ==                          Project Data (PCA) |   0 /  10 | 
  ==                          Recover Data (PCA) |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  80 / 100 | 
  == 
  ```

- 在使用PCA算法前，先利用 featureNormalize.m 函数正则化特征，在命令行输入

  ```matlab
  % Before running PCA, it is important to first normalize X
  [X_norm, mu, ~] = featureNormalize(X);
  
  % Run PCA
  [U, S] = pca(X_norm);
  
  % Draw the eigenvectors centered at mean of data. These lines show the directions of maximum variations in the dataset.
  hold on;
  drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
  drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
  hold off;
  
  fprintf('Top eigenvector U(:,1) = %f %f \n', U(1,1), U(2,1));
  ```

![image-20200723122235932](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex7/image-20200723122235932.png)



#### 2.3  用PCA算法降维 Dimensionality reduction with PCA



- 打开 projectData.m 函数，做如下更改

  ```matlab
  function Z = projectData(X, U, K)
  %PROJECTDATA Computes the reduced data representation when projecting only 
  %on to the top k eigenvectors
  %   Z = projectData(X, U, K) computes the projection of 
  %   the normalized inputs X into the reduced dimensional space spanned by
  %   the first K columns of U. It returns the projected examples in Z.
  %
  
  % You need to return the following variables correctly.
  Z = zeros(size(X, 1), K);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the projection of the data using only the top K 
  %               eigenvectors in U (first K columns). 
  %               For the i-th example X(i,:), the projection on to the k-th 
  %               eigenvector is given as follows:
  %                    x = X(i, :)';
  %                    projection_k = x' * U(:, k);
  %
  
  U_reduce = U(:, 1:K);
  Z = X * U_reduce;
  
  % =============================================================
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | K-Means Clustering and PCA...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==            Find Closest Centroids (k-Means) |  30 /  30 | Nice work!
  ==            Compute Centroid Means (k-Means) |  30 /  30 | Nice work!
  ==                                         PCA |  20 /  20 | Nice work!
  ==                          Project Data (PCA) |  10 /  10 | Nice work!
  ==                          Recover Data (PCA) |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  90 / 100 | 
  == 
  ```



#### 2.4  重建数据的近似值 Reconstructing an approximation of the data



- 打开 recoverData.m 函数，做如下更改

  ```matlab
  function X_rec = recoverData(Z, U, K)
  %RECOVERDATA Recovers an approximation of the original data when using the 
  %projected data
  %   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
  %   original data that has been reduced to K dimensions. It returns the
  %   approximate reconstruction in X_rec.
  %
  
  % You need to return the following variables correctly.
  X_rec = zeros(size(Z, 1), size(U, 1));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the approximation of the data by projecting back
  %               onto the original space using the top K eigenvectors in U.
  %
  %               For the i-th example Z(i,:), the (approximate)
  %               recovered data for dimension j is given as follows:
  %                    v = Z(i, :)';
  %                    recovered_j = v' * U(j, 1:K)';
  %
  %               Notice that U(j, 1:K) is a row vector.
  %               
  
  U_reduce = U(:, 1:K);
  X_rec = Z * U_reduce';
  
  % =============================================================
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | K-Means Clustering and PCA...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==            Find Closest Centroids (k-Means) |  30 /  30 | Nice work!
  ==            Compute Centroid Means (k-Means) |  30 /  30 | Nice work!
  ==                                         PCA |  20 /  20 | Nice work!
  ==                          Project Data (PCA) |  10 /  10 | Nice work!
  ==                          Recover Data (PCA) |  10 /  10 | Nice work!
  ==                                   --------------------------------
  ==                                             | 100 / 100 | 
  == 
  ```

- 可视化降维投影

  ```matlab
  %  Plot the normalized dataset (returned from pca)
  plot(X_norm(:, 1), X_norm(:, 2), 'bo');
  axis([-4 3 -4 3]); axis square
  %  Draw lines connecting the projected points to the original points
  hold on;
  plot(X_rec(:, 1), X_rec(:, 2), 'ro');
  for i = 1:size(X_norm, 1)
      drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
  end
  hold off
  ```

![image-20200723123622367](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex7/image-20200723123622367.png)

