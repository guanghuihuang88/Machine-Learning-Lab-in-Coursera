# ex8  异常检测和推荐系统 Anomaly Detection and Recommender Systems

[toc]

### 1.  异常检测 Anomaly Detection



- 可视化训练集

  ```matlab
  % The following command loads the dataset. You should now have the variables X, Xval, yval in your environment
  load('ex8data1.mat');
  
  % Visualize the example dataset
  plot(X(:, 1), X(:, 2), 'bx');
  axis([0 30 0 30]);
  xlabel('Latency (ms)');
  ylabel('Throughput (mb/s)');
  ```

  ![image-20200730105659879](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex8/image-20200730105659879.png)



#### 1.2  为高斯分布训练参数 Estimating parameters for a Gaussian


- 打开 estimateGaussian.m 函数，做如下更改

  ```matlab
  function [mu sigma2] = estimateGaussian(X)
  %ESTIMATEGAUSSIAN This function estimates the parameters of a 
  %Gaussian distribution using the data in X
  %   [mu sigma2] = estimateGaussian(X), 
  %   The input X is the dataset with each n-dimensional data point in one row
  %   The output is an n-dimensional vector mu, the mean of the data set
  %   and the variances sigma^2, an n x 1 vector
  % 
  
  % Useful variables
  [m, n] = size(X);
  
  % You should return these values correctly
  mu = zeros(n, 1);
  sigma2 = zeros(n, 1);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the mean of the data and the variances
  %               In particular, mu(i) should contain the mean of
  %               the data for the i-th feature and sigma2(i)
  %               should contain variance of the i-th feature.
  %
  
  for i = 1:size(X, 2)
      mu(i) = mean(X(:, i));
  end
  sigma2 = ((m-1)/m) * var(X);
  
  % =============================================================
  
  
  end
  ```
  
- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Anomaly Detection and Recommender Systems...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                Estimate Gaussian Parameters |  15 /  15 | Nice work!
  ==                            Select Threshold |   0 /  15 | 
  ==                Collaborative Filtering Cost |   0 /  20 | 
  ==            Collaborative Filtering Gradient |   0 /  30 | 
  ==                            Regularized Cost |   0 /  10 | 
  ==                        Regularized Gradient |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  15 / 100 | 
  == 
  ```

- 检验函数正确性

  ```matlab
  %  Estimate mu and sigma2
  [mu, sigma2] = estimateGaussian(X);
  
  %  Returns the density of the multivariate normal at each data point (row) of X
  p = multivariateGaussian(X, mu, sigma2);
  
  %  Visualize the fit
  visualizeFit(X,  mu, sigma2);
  xlabel('Latency (ms)');
  ylabel('Throughput (mb/s)');
  ```

![image-20200730121457996](插图/ex8/image-20200730121457996.png)




#### 1.3  选择阈值 Selecting the threshold 

- 打开 selectThreshold.m 函数，做如下更改

  ```matlab
  function [bestEpsilon bestF1] = selectThreshold(yval, pval)
  %SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
  %outliers
  %   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
  %   threshold to use for selecting outliers based on the results from a
  %   validation set (pval) and the ground truth (yval).
  %
  
  bestEpsilon = 0;
  bestF1 = 0;
  F1 = 0;
  
  stepsize = (max(pval) - min(pval)) / 1000;
  for epsilon = min(pval):stepsize:max(pval)
      
      % ====================== YOUR CODE HERE ======================
      % Instructions: Compute the F1 score of choosing epsilon as the
      %               threshold and place the value in F1. The code at the
      %               end of the loop will compare the F1 score for this
      %               choice of epsilon and set it to be the best epsilon if
      %               it is better than the current choice of epsilon.
      %               
      % Note: You can use predictions = (pval < epsilon) to get a binary vector
      %       of 0's and 1's of the outlier predictions
  
      predictions = (pval < epsilon);
      tp = sum(predictions == 1 & yval == 1);
      fp = sum(predictions == 1 & yval == 0);
      fn = sum(predictions == 0 & yval == 1);
      prec = tp / (tp + fp);
      rec = tp / (tp + fn);
      F1 = (2 * prec * rec) / (prec + rec);
  
      % =============================================================
  
      if F1 > bestF1
         bestF1 = F1;
         bestEpsilon = epsilon;
      end
  end
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Anomaly Detection and Recommender Systems...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                Estimate Gaussian Parameters |  15 /  15 | Nice work!
  ==                            Select Threshold |  15 /  15 | Nice work!
  ==                Collaborative Filtering Cost |   0 /  20 | 
  ==            Collaborative Filtering Gradient |   0 /  30 | 
  ==                            Regularized Cost |   0 /  10 | 
  ==                        Regularized Gradient |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  30 / 100 | 
  == 
  ```

- 检验函数正确性

  ```matlab
  pval = multivariateGaussian(Xval, mu, sigma2);
  
  [epsilon, F1] = selectThreshold(yval, pval);
  fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
  fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
  
  %  Find the outliers in the training set and plot the
  outliers = find(p < epsilon);
  
  %  Visualize the fit
  visualizeFit(X,  mu, sigma2);
  xlabel('Latency (ms)');
  ylabel('Throughput (mb/s)');
  %  Draw a red circle around those outliers
  hold on
  plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
  hold off
  ```

  ![image-20200731133116154](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex8/image-20200731133116154.png)



### 2.  推荐系统 Recommender Systems



#### 2.1  电影评分数据集 Movie ratings dataset



- 可视化电影评分数据集

  ```matlab
  % Load data
  load('ex8_movies.mat');
  
  % From the matrix, we can compute statistics like average rating.
  fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', mean(Y(1, R(1, :))));
  %  We can "visualize" the ratings matrix by plotting it with imagesc
  imagesc(Y);
  ylabel('Movies');
  xlabel('Users');
  ```
  
  ![image-20200723115420950](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex8/untitled.png)
  
  

#### 2.2  协同过滤算法 Collaborative filtering learning algorithm



##### 2.2.1  协同过滤代价函数Collaborative filtering cost function

- 打开 cofiCostFunc.m 函数，做如下更改

  ```matlab
  function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                    num_features, lambda)
  %COFICOSTFUNC Collaborative filtering cost function
  %   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
  %   num_features, lambda) returns the cost and gradient for the
  %   collaborative filtering problem.
  %
  
  % Unfold the U and W matrices from params
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), ...
                  num_users, num_features);
  
              
  % You need to return the following values correctly
  J = 0;
  X_grad = zeros(size(X));
  Theta_grad = zeros(size(Theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost function and gradient for collaborative
  %               filtering. Concretely, you should first implement the cost
  %               function (without regularization) and make sure it is
  %               matches our costs. After that, you should implement the 
  %               gradient and use the checkCostFunction routine to check
  %               that the gradient is correct. Finally, you should implement
  %               regularization.
  %
  % Notes: X - num_movies  x num_features matrix of movie features
  %        Theta - num_users  x num_features matrix of user features
  %        Y - num_movies x num_users matrix of user ratings of movies
  %        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
  %            i-th movie was rated by the j-th user
  %
  % You should set the following variables correctly:
  %
  %        X_grad - num_movies x num_features matrix, containing the 
  %                 partial derivatives w.r.t. to each element of X
  %        Theta_grad - num_users x num_features matrix, containing the 
  %                     partial derivatives w.r.t. to each element of Theta
  %
  
  J = (1/2) * sum(sum((R .* ((X * Theta' - Y) .^ 2))));
  
  % =============================================================
  
  grad = [X_grad(:); Theta_grad(:)];
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Anomaly Detection and Recommender Systems...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                Estimate Gaussian Parameters |  15 /  15 | Nice work!
  ==                            Select Threshold |  15 /  15 | Nice work!
  ==                Collaborative Filtering Cost |  20 /  20 | Nice work!
  ==            Collaborative Filtering Gradient |   0 /  30 | 
  ==                            Regularized Cost |   0 /  10 | 
  ==                        Regularized Gradient |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  50 / 100 | 
  == 
  ```

- 检验代价函数

  ```matlab
  %  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
  load('ex8_movieParams.mat');
  
  %  Reduce the data set size so that this runs faster
  num_users = 4; num_movies = 5; num_features = 3;
  X = X(1:num_movies, 1:num_features);
  Theta = Theta(1:num_users, 1:num_features);
  Y = Y(1:num_movies, 1:num_users);
  R = R(1:num_movies, 1:num_users);
  
  %  Evaluate cost function
  J = cofiCostFunc([X(:); Theta(:)],  Y, R, num_users, num_movies,num_features, 0);
  fprintf('Cost at loaded parameters: %f ',J);
  ```



##### 2.2.2  协同过滤梯度 Collaborative filtering gradient

- 打开 cofiCostFunc.m 函数，做如下更改

  ```matlab
  function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                    num_features, lambda)
  %COFICOSTFUNC Collaborative filtering cost function
  %   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
  %   num_features, lambda) returns the cost and gradient for the
  %   collaborative filtering problem.
  %
  
  % Unfold the U and W matrices from params
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), ...
                  num_users, num_features);
  
              
  % You need to return the following values correctly
  J = 0;
  X_grad = zeros(size(X));
  Theta_grad = zeros(size(Theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost function and gradient for collaborative
  %               filtering. Concretely, you should first implement the cost
  %               function (without regularization) and make sure it is
  %               matches our costs. After that, you should implement the 
  %               gradient and use the checkCostFunction routine to check
  %               that the gradient is correct. Finally, you should implement
  %               regularization.
  %
  % Notes: X - num_movies  x num_features matrix of movie features
  %        Theta - num_users  x num_features matrix of user features
  %        Y - num_movies x num_users matrix of user ratings of movies
  %        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
  %            i-th movie was rated by the j-th user
  %
  % You should set the following variables correctly:
  %
  %        X_grad - num_movies x num_features matrix, containing the 
  %                 partial derivatives w.r.t. to each element of X
  %        Theta_grad - num_users x num_features matrix, containing the 
  %                     partial derivatives w.r.t. to each element of Theta
  %
  
  J = (1/2) * sum(sum((R .* ((X * Theta' - Y) .^ 2))));
  
  X_grad = (R .* (X * Theta' - Y)) * Theta;
  Theta_grad = (R .* (X * Theta' - Y))' * X;
  
  % =============================================================
  
  grad = [X_grad(:); Theta_grad(:)];
  
  end
  ```

- 在命令行提交

  ```matlab
  是不是想输入:
  >> submit
  == Submitting solutions | Anomaly Detection and Recommender Systems...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                Estimate Gaussian Parameters |  15 /  15 | Nice work!
  ==                            Select Threshold |  15 /  15 | Nice work!
  ==                Collaborative Filtering Cost |  20 /  20 | Nice work!
  ==            Collaborative Filtering Gradient |  30 /  30 | Nice work!
  ==                            Regularized Cost |   0 /  10 | 
  ==                        Regularized Gradient |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  80 / 100 | 
  == 
  ```



##### 2.2.3  正则化代价函数 Regularized cost function

- 打开 cofiCostFunc.m 函数，做如下更改

  ```matlab
  function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                    num_features, lambda)
  %COFICOSTFUNC Collaborative filtering cost function
  %   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
  %   num_features, lambda) returns the cost and gradient for the
  %   collaborative filtering problem.
  %
  
  % Unfold the U and W matrices from params
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), ...
                  num_users, num_features);
  
              
  % You need to return the following values correctly
  J = 0;
  X_grad = zeros(size(X));
  Theta_grad = zeros(size(Theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost function and gradient for collaborative
  %               filtering. Concretely, you should first implement the cost
  %               function (without regularization) and make sure it is
  %               matches our costs. After that, you should implement the 
  %               gradient and use the checkCostFunction routine to check
  %               that the gradient is correct. Finally, you should implement
  %               regularization.
  %
  % Notes: X - num_movies  x num_features matrix of movie features
  %        Theta - num_users  x num_features matrix of user features
  %        Y - num_movies x num_users matrix of user ratings of movies
  %        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
  %            i-th movie was rated by the j-th user
  %
  % You should set the following variables correctly:
  %
  %        X_grad - num_movies x num_features matrix, containing the 
  %                 partial derivatives w.r.t. to each element of X
  %        Theta_grad - num_users x num_features matrix, containing the 
  %                     partial derivatives w.r.t. to each element of Theta
  %
  
  J = (1/2) * sum(sum((R .* ((X * Theta' - Y) .^ 2)))) + (lambda/2) * sum(sum(Theta .^ 2)) + (lambda/2) * sum(sum(X .^ 2));
  
  X_grad = (R .* (X * Theta' - Y)) * Theta;
  Theta_grad = (R .* (X * Theta' - Y))' * X;
  
  % =============================================================
  
  grad = [X_grad(:); Theta_grad(:)];
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Anomaly Detection and Recommender Systems...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                Estimate Gaussian Parameters |  15 /  15 | Nice work!
  ==                            Select Threshold |  15 /  15 | Nice work!
  ==                Collaborative Filtering Cost |  20 /  20 | Nice work!
  ==            Collaborative Filtering Gradient |  30 /  30 | Nice work!
  ==                            Regularized Cost |  10 /  10 | Nice work!
  ==                        Regularized Gradient |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  90 / 100 | 
  == 
  ```




##### 2.2.4  正则化梯度 Regularized gradient

- 打开 cofiCostFunc.m 函数，做如下更改

  ```matlab
  function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                    num_features, lambda)
  %COFICOSTFUNC Collaborative filtering cost function
  %   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
  %   num_features, lambda) returns the cost and gradient for the
  %   collaborative filtering problem.
  %
  
  % Unfold the U and W matrices from params
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), ...
                  num_users, num_features);
  
              
  % You need to return the following values correctly
  J = 0;
  X_grad = zeros(size(X));
  Theta_grad = zeros(size(Theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost function and gradient for collaborative
  %               filtering. Concretely, you should first implement the cost
  %               function (without regularization) and make sure it is
  %               matches our costs. After that, you should implement the 
  %               gradient and use the checkCostFunction routine to check
  %               that the gradient is correct. Finally, you should implement
  %               regularization.
  %
  % Notes: X - num_movies  x num_features matrix of movie features
  %        Theta - num_users  x num_features matrix of user features
  %        Y - num_movies x num_users matrix of user ratings of movies
  %        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
  %            i-th movie was rated by the j-th user
  %
  % You should set the following variables correctly:
  %
  %        X_grad - num_movies x num_features matrix, containing the 
  %                 partial derivatives w.r.t. to each element of X
  %        Theta_grad - num_users x num_features matrix, containing the 
  %                     partial derivatives w.r.t. to each element of Theta
  %
  
  J = (1/2) * sum(sum((R .* ((X * Theta' - Y) .^ 2)))) + (lambda/2) * sum(sum(Theta .^ 2)) + (lambda/2) * sum(sum(X .^ 2));
  
  X_grad = (R .* (X * Theta' - Y)) * Theta + lambda * X;
  Theta_grad = (R .* (X * Theta' - Y))' * X + lambda * Theta;
  
  % =============================================================
  
  grad = [X_grad(:); Theta_grad(:)];
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Anomaly Detection and Recommender Systems...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                Estimate Gaussian Parameters |  15 /  15 | Nice work!
  ==                            Select Threshold |  15 /  15 | Nice work!
  ==                Collaborative Filtering Cost |  20 /  20 | Nice work!
  ==            Collaborative Filtering Gradient |  30 /  30 | Nice work!
  ==                            Regularized Cost |  10 /  10 | Nice work!
  ==                        Regularized Gradient |  10 /  10 | Nice work!
  ==                                   --------------------------------
  ==                                             | 100 / 100 | 
  == 
  ```

