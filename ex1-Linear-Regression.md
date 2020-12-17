## ex1： 线性回归  Linear Regression

[toc]

### 1. 热身训练  A simple MATLAB function

- 打开 warmUpExercise.m 函数，做如下更改

  ```matlab
  function A = warmUpExercise()
  %WARMUPEXERCISE Example function in octave
  %   A = WARMUPEXERCISE() is an example function that returns the 5x5 identity matrix
  
  A = [];
  % ============= YOUR CODE HERE ==============
  % Instructions: Return the 5x5 identity matrix 
  %               In octave, we return values by defining which variables
  %               represent the return values (at the top of the file)
  %               and then set them accordingly. 
  
  A = eye(5);
  
  % ===========================================
  
  end
  ```

- 在命令行输出

  ```matlab
  warmUpExercise()
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Linear Regression with Multiple Variables...
  Login (email address): 
  guanghuihuang88@gmail.com
  Token: 
  td8ycXz14JquWXiU
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                            Warm-up Exercise |  10 /  10 | Nice work!
  ==           Computing Cost (for One Variable) |   0 /  40 | 
  ==         Gradient Descent (for One Variable) |   0 /  50 | 
  ==                       Feature Normalization |   0 /   0 | 
  ==     Computing Cost (for Multiple Variables) |   0 /   0 | 
  ==   Gradient Descent (for Multiple Variables) |   0 /   0 | 
  ==                            Normal Equations |   0 /   0 | 
  ==                                   --------------------------------
  ==                                             |  10 / 100 | 
  ```





### 2. 单变量线性回归  Linear regression with one variable



#### 2.1 可视化数据  Plotting the data

- 在命令行上传训练集

  ```matlab
  >> data = load('ex1data1.txt');
  >> X = data(:, 1);
  >> y = data(:, 2);
  ```

- 打开 plotData.m 函数，做如下更改

  ```matlab
  function plotData(x, y)
  %PLOTDATA Plots the data points x and y into a new figure 
  %   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
  %   population and profit.
  
  figure; % open a new figure window
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Plot the training data into a figure using the 
  %               "figure" and "plot" commands. Set the axes labels using
  %               the "xlabel" and "ylabel" commands. Assume the 
  %               population and revenue data have been passed in
  %               as the x and y arguments of this function.
  %
  % Hint: You can use the 'rx' option with plot to have the markers
  %       appear as red crosses. Furthermore, you can make the
  %       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
  
  plot(x, y, 'rx', 'MarkerSize', 10);
  ylabel('Profit in $10,000s'); 
  xlabel('Population of City in 10,000s'); 
  
  % ============================================================
  
  end
  ```

- 在命令行输出

  ```matlab
  plotData(X,y)
  ```

  ![image-20200722131722887](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex1/image-20200722131722887.png)
  
  

#### 2.2 梯度下降算法  Gradient Descent

- 配置梯度下降算法需要的参数

  ```matlab
  >> m = length(X)
  
  m =
  
      97
  
  >> X = [ones(m, 1), data(:, 1)];
  >> theta = zeros(2, 1);
  >> iterations = 1500;
  >> alpha = 0.01;
  ```

- 计算代价函数：

  打开 computeCost.m 函数，做如下更改

  ```matlab
  function J = computeCost(X, y, theta)
  %COMPUTECOST Compute cost for linear regression
  %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
  %   parameter for linear regression to fit the data points in X and y
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  % You need to return the following variables correctly 
  J = 0;
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost of a particular choice of theta
  %               You should set J to the cost.
  
  J = (1/m) * ((1/2) * power(X * theta - y, 2))' * ones(m, 1);
  
  % =========================================================================
  
  end
  ```

- 检验函数正确性：

  在命令行输出

  ```matlab
  >> computeCost(X, y, theta)
  
  ans =
  
     32.0727
  
  >> computeCost(X, y,[-1; 2])
  
  ans =
  
     54.2425
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Linear Regression with Multiple Variables...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                            Warm-up Exercise |  10 /  10 | Nice work!
  ==           Computing Cost (for One Variable) |  40 /  40 | Nice work!
  ==         Gradient Descent (for One Variable) |   0 /  50 | 
  ==                       Feature Normalization |   0 /   0 | 
  ==     Computing Cost (for Multiple Variables) |   0 /   0 | 
  ==   Gradient Descent (for Multiple Variables) |   0 /   0 | 
  ==                            Normal Equations |   0 /   0 | 
  ==                                   --------------------------------
  ==                                             |  50 / 100 | 
  ```

- 批量梯度下降：

  打开 gradientDescent.m 函数，做如下更改

  ```matlab
  function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
  %   taking num_iters gradient steps with learning rate alpha
  
  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  
  for iter = 1:num_iters
  
      % ====================== YOUR CODE HERE ======================
      % Instructions: Perform a single gradient step on the parameter vector
      %               theta. 
      %
      % Hint: While debugging, it can be useful to print out the values
      %       of the cost function (computeCost) and gradient here.
      %
      
      temp1 = theta(1) - alpha * (1/m) * ((X * theta - y)' * ones(m, 1));
      temp2 = theta(2) - alpha * (1/m) * ((X * theta - y)' * X(:, 2));
      theta(1) = temp1;
      theta(2) = temp2;
  
      % ============================================================
  
      % Save the cost J in every iteration    
      J_history(iter) = computeCost(X, y, theta);
  
  end
  
  end
  ```

- 检验函数正确性：

  在命令行输出

  ```matlab
  % Run gradient descent:
  % Compute theta
  theta = gradientDescent(X, y, theta, alpha, iterations);
  
  % Print theta to screen
  % Display gradient descent's result
  fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))
  
  % Plot the linear fit
  hold on; % keep previous plot visible
  plot(X(:,2), X*theta, '-')
  legend('Training data', 'Linear regression')
  hold off % don't overlay any more plots on this figure
  
  % Predict values for population sizes of 35,000 and 70,000
  predict1 = [1, 3.5] *theta;
  fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);
  predict2 = [1, 7] * theta;
  fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);
  ```

![image-20200722132002617](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex1/image-20200722132002617.png)

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Linear Regression with Multiple Variables...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                            Warm-up Exercise |  10 /  10 | Nice work!
  ==           Computing Cost (for One Variable) |  40 /  40 | Nice work!
  ==         Gradient Descent (for One Variable) |  50 /  50 | Nice work!
  ==                       Feature Normalization |   0 /   0 | 
  ==     Computing Cost (for Multiple Variables) |   0 /   0 | 
  ==   Gradient Descent (for Multiple Variables) |   0 /   0 | 
  ==                            Normal Equations |   0 /   0 | 
  ==                                   --------------------------------
  ==                                             | 100 / 100 | 
  ```

  

### 3. 多变量线性回归  Linear regression with multiple variables

- 在命令行上传训练集

  ```matlab
  >> data = load('ex1data2.txt');
  >> X = data(:, 1:2);
  >> y = data(:, 3);
  >> m = length(y);
  ```

- 打印前十个数据稍作了解

  ```matlab
  >> fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
   x = [2104 3], y = 399900 
   x = [1600 3], y = 329900 
   x = [2400 3], y = 369000 
   x = [1416 2], y = 232000 
   x = [3000 4], y = 539900 
   x = [1985 4], y = 299900 
   x = [1534 3], y = 314900 
   x = [1427 3], y = 198999 
   x = [1380 3], y = 212000 
   x = [1494 3], y = 242500 
  ```

  


#### 3.1 特征缩放  Feature Scaling（Normalization）

- 特征缩放（对训练集 X 进行放缩处理，加快之后梯度下降的速度）：

  打开 featureNormalize.m 函数，做如下更改

  ```matlab
  function [X_norm, mu, sigma] = featureNormalize(X)
  %FEATURENORMALIZE Normalizes the features in X 
  %   FEATURENORMALIZE(X) returns a normalized version of X where
  %   the mean value of each feature is 0 and the standard deviation
  %   is 1. This is often a good preprocessing step to do when
  %   working with learning algorithms.
  
  % You need to set these values correctly
  X_norm = X;
  mu = zeros(1, size(X, 2));
  sigma = zeros(1, size(X, 2));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: First, for each feature dimension, compute the mean
  %               of the feature and subtract it from the dataset,
  %               storing the mean value in mu. Next, compute the 
  %               standard deviation of each feature and divide
  %               each feature by it's standard deviation, storing
  %               the standard deviation in sigma. 
  %
  %               Note that X is a matrix where each column is a 
  %               feature and each row is an example. You need 
  %               to perform the normalization separately for 
  %               each feature. 
  %
  % Hint: You might find the 'mean' and 'std' functions useful.
  %       
  
  m = size(X, 1);
  for i  = 1 : size(X, 2)
      mu(i) = mean(X(:, i));
      sigma(i) = std(X(:, i));
      X_norm(:, i) = (X(:, i) - mu(i) * ones(m, 1)) / sigma(i);
  end
  
  % ============================================================
  
  end
  ```

- 检验函数正确性：

  在命令行输出

  ```matlab
  >> [X, mu, sigma] = featureNormalize(X);
  ```

  

#### 3.2 梯度下降算法  Gradient Descent

- 配置梯度下降算法需要的参数

  ```matlab
  >> X = [ones(m, 1) X];
  >> alpha = 0.1;
  >> num_iters = 400;
  >> theta = zeros(3, 1);
  ```

- 代价函数：

  打开 computeCostMulti.m 函数，做如下更改

  ```matlab
  function J = computeCost(X, y, theta)
  %COMPUTECOST Compute cost for linear regression
  %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
  %   parameter for linear regression to fit the data points in X and y
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  % You need to return the following variables correctly 
  J = 0;
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost of a particular choice of theta
  %               You should set J to the cost.
  
  J = (1/m) * ((1/2) * power(X * theta - y, 2))' * ones(m, 1);
  
  % =========================================================================
  
  end
  ```

- 批量梯度下降：

  打开 gradientDescentMulti.m 函数，做如下更改

  ```matlab
  function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
  %   taking num_iters gradient steps with learning rate alpha
  
  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  
  for iter = 1:num_iters
  
      % ====================== YOUR CODE HERE ======================
      % Instructions: Perform a single gradient step on the parameter vector
      %               theta. 
      %
      % Hint: While debugging, it can be useful to print out the values
      %       of the cost function (computeCostMulti) and gradient here.
      %
      
      temp = zeros(size(X, 2), 1);
      for i = 1 : size(X, 2)
          temp(i) = theta(i) - alpha * (1/m) * ((X * theta - y)' * X(:, i));
      end
      for i = 1 : size(X, 2)
          theta(i) = temp(i);
      end
      
      % ============================================================
  
      % Save the cost J in every iteration    
      J_history(iter) = computeCostMulti(X, y, theta);
  
  end
  
  end
  ```

- 检验函数正确性：

  在命令行输出

  ```matlab
  >> [theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  
  theta =
  
     1.0e+05 *
  
      3.4041
      1.1063
     -0.0665
  
  % Display gradient descent's result
  >> fprintf('Theta computed from gradient descent:\n%f,\n%f,\n%f',theta(1),theta(2),theta(3))
  
  Theta computed from gradient descent:
  340412.659574,
  110631.050279,
  -6649.474271
  ```

- 用训练好的 参数theta 预测 1650 sq-ft, 3 个房间的房价

  即 x = (1, 1650, 3)，price = x * theta

  注意，使用的训练集 X 做过放缩处理，因此 x 也需要经过同样的放缩处理

  x = [1,  (1650-mu(1)) / sigma(1),  (3-mu(2)) / sigma(2)]

  ```matlab
  >> price = [1, (1650-mu(1)) / sigma(1), (3-mu(2)) / sigma(2)] * theta
  
  price =
  
     2.9308e+05
  
  >> fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);
  
  Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):
   $293081.464335
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Linear Regression with Multiple Variables...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                            Warm-up Exercise |  10 /  10 | Nice work!
  ==           Computing Cost (for One Variable) |  40 /  40 | Nice work!
  ==         Gradient Descent (for One Variable) |  50 /  50 | Nice work!
  ==                       Feature Normalization |   0 /   0 | Nice work!
  ==     Computing Cost (for Multiple Variables) |   0 /   0 | Nice work!
  ==   Gradient Descent (for Multiple Variables) |   0 /   0 | Nice work!
  ==                            Normal Equations |   0 /   0 | 
  ==                                   --------------------------------
  ==                                             | 100 / 100 | 
  ```

- 修改学习率，观察梯度下降速度

  ```matlab
  >> alpha = 1;
  >> [~, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
  >> plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
  >> xlabel('Number of iterations');
  >> ylabel('Cost J');
  >> hold on
  >> alpha = 0.1;
  >> [~, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
  >> plot(1:num_iters, J_history, '-r', 'LineWidth', 2);
  >> alpha = 0.01;
  >> [~, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
  >> plot(1:num_iters, J_history, '-k', 'LineWidth', 2);
  >> alpha = 0.001;
  >> [~, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
  >> plot(1:num_iters, J_history, '-g', 'LineWidth', 2);
  >> hold off
  ```

  

#### 3.3 正规方程  Normal Equations

- 配置正规方程需要的参数

  ```matlab
  >> data = csvread('ex1data2.txt');
  >> X = data(:, 1:2);
  >> y = data(:, 3);
  >> m = length(y);
  >> X = [ones(m, 1) X];
  ```

- 正规方程：

  打开 normalEqn.m 函数，做如下更改

  ```matlab
  function [theta] = normalEqn(X, y)
  %NORMALEQN Computes the closed-form solution to linear regression 
  %   NORMALEQN(X,y) computes the closed-form solution to linear 
  %   regression using the normal equations.
  
  theta = zeros(size(X, 2), 1);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Complete the code to compute the closed form solution
  %               to linear regression and put the result in theta.
  %
  
  % ---------------------- Sample Solution ----------------------
  
  theta = inv(X' * X) * X' * y;
  
  % -------------------------------------------------------------
  
  
  % ============================================================
  
  end
  
  ```

- 检验函数正确性：

  在命令行输出

  ```matlab
  >> theta = normalEqn(X, y)
  
  theta =
  
     1.0e+04 *
  
      8.9598
      0.0139
     -0.8738
  
  >> price = [1, 1650, 3] * theta
  
  price =
  
     2.9308e+05
  
  >> fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price);
  Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):
   $293081.464335
  ```

  - 注意：theta 计算出的结果与梯度下降算法不同，这是因为梯度下降算法的训练集做了放缩处理，容易看出，当预测 1650 sq-ft, 3 个房间的房价时，无需放缩处理，且结果与梯度下降算法结果相同，为$293081.464335

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Linear Regression with Multiple Variables...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                            Warm-up Exercise |  10 /  10 | Nice work!
  ==           Computing Cost (for One Variable) |  40 /  40 | Nice work!
  ==         Gradient Descent (for One Variable) |  50 /  50 | Nice work!
  ==                       Feature Normalization |   0 /   0 | Nice work!
  ==     Computing Cost (for Multiple Variables) |   0 /   0 | Nice work!
  ==   Gradient Descent (for Multiple Variables) |   0 /   0 | Nice work!
  ==                            Normal Equations |   0 /   0 | Nice work!
  ==                                   --------------------------------
  ==                                             | 100 / 100 | 
  ```