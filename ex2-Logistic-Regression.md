## ex2: 逻辑回归  Logistic Regression

[toc]

### 1. 逻辑回归  Logistic Regression

- 上传训练集

  ```matlab
  >> data = load('ex2data1.txt');
  >> X = data(:, [1, 2]);
  >> y = data(:, 3);
  ```



#### 1.1 可视化训练集  Visualizing the data

- 打开plotData.m函数，做如下更改

  ```matlab
  function plotData(X, y)
  %PLOTDATA Plots the data points X and y into a new figure 
  %   PLOTDATA(x,y) plots the data points with + for the positive examples
  %   and o for the negative examples. X is assumed to be a Mx2 matrix.
  
  % Create New Figure
  figure; hold on;
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Plot the positive and negative examples on a
  %               2D plot, using the option 'k+' for the positive
  %               examples and 'ko' for the negative examples.
  %
      
  % Find Indices of Positive and Negative Examples
  pos = find(y==1); neg = find(y == 0);
  % Plot Examples
  plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
  plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);
  xlabel('Exam 1 score');
  ylabel('Exam 2 score');
  legend('Admitted', 'Not admitted');
  
  % =========================================================================
  
  hold off;
  
  end
  ```
  
  ![image-20200611232017042](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex2/image-20200611232017042.png)

#### 1.2 代价函数  CostFunction

- 打开 sigmoid.m 函数，做如下更改

  ```matlab
  function g = sigmoid(z)
  %SIGMOID Compute sigmoid functionG
  %   g = SIGMOID(z) computes the sigmoid of z.
  
  % You need to return the following variables correctly 
  g = zeros(size(z));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
  %               vector or scalar).
  
  for i = 1:size(g, 1)
      for j = 1:size(g, 2)
          g(i, j) = 1 / (1 + exp(-z(i, j)));
  	end
  end
  
  % =============================================================
  
  end
  ```

- 验证sigmoid.m函数

  在命令行输入

  ```matlab
  >> sigmoid(0)
  
  ans =
  
      0.5000
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
  ==                            Sigmoid Function |   5 /   5 | Nice work!
  ==                    Logistic Regression Cost |   0 /  30 | 
  ==                Logistic Regression Gradient |   0 /  30 | 
  ==                                     Predict |   0 /   5 | 
  ==        Regularized Logistic Regression Cost |   0 /  15 | 
  ==    Regularized Logistic Regression Gradient |   0 /  15 | 
  ==                                   --------------------------------
  ==                                             |   5 / 100 | 
  ```



- 对训练集进行加工

  ```matlab
  >> [m, n] = size(X);
  >> X = [ones(m, 1) X];
  >> initial_theta = zeros(n + 1, 1);
  ```

- 打开 costFunction.m 函数，做如下更改

  ```matlab
  function [J, grad] = costFunction(theta, X, y)
  %COSTFUNCTION Compute cost and gradient for logistic regression
  %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
  %   parameter for logistic regression and the gradient of the cost
  %   w.r.t. to the parameters.
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost of a particular choice of theta.
  %               You should set J to the cost.
  %               Compute the partial derivatives and set grad to the partial
  %               derivatives of the cost w.r.t. each parameter in theta
  %
  % Note: grad should have the same dimensions as theta
  %
  
  J = (-1 / m) * (y' * log(sigmoid(X*theta)) + (1 - y)' * log(1 - sigmoid(X*theta)));
  
  grad = (1/m) * ((sigmoid(X*theta) - y)' * X)';
  
  % =============================================================
  
  end
  ```

- 验证 costFunction.m 函数

  在命令行输入

  ```matlab
  >> [cost, grad] = costFunction(initial_theta, X, y);
  >> fprintf('Cost at initial theta (zeros): %f\n', cost);
  >> disp('Gradient at initial theta (zeros):'); disp(grad);
  
  Cost at initial theta (zeros): 0.693147
  Gradient at initial theta (zeros):
     -0.1000
    -12.0092
    -11.2628
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Logistic Regression...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                            Sigmoid Function |   5 /   5 | Nice work!
  ==                    Logistic Regression Cost |  30 /  30 | Nice work!
  ==                Logistic Regression Gradient |  30 /  30 | Nice work!
  ==                                     Predict |   0 /   5 | 
  ==        Regularized Logistic Regression Cost |   0 /  15 | 
  ==    Regularized Logistic Regression Gradient |   0 /  15 | 
  ==                                   --------------------------------
  ==                                             |  65 / 100 | 
  ==  
  ```



#### 1.3 用fminucs替代梯度下降算法求解最优化问题

- 需要自己编写好代价函数costFunction

```matlab
%  Set options for fminunc
>> options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);


%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
>> [theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

Local minimum found.

Optimization completed because the size of the gradient is less than
the value of the optimality tolerance.

<stopping criteria details>


% Print theta
>> fprintf('Cost at theta found by fminunc: %f\n', cost);
>> disp('theta:');disp(theta);
Cost at theta found by fminunc: 0.203498
theta:
  -25.1613
    0.2062
    0.2015


% Plot Boundary
plotDecisionBoundary(theta, X, y);
% Add some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;
```

![image-20200611231957524](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex2/image-20200611231957524.png)

- 打开 predict.m 函数，做如下更改

  ```matlab
  function p = predict(theta, X)
  %PREDICT Predict whether the label is 0 or 1 using learned logistic 
  %regression parameters theta
  %   p = PREDICT(theta, X) computes the predictions for X using a 
  %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
  
  m = size(X, 1); % Number of training examples
  
  % You need to return the following variables correctly
  p = zeros(m, 1);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Complete the following code to make predictions using
  %               your learned logistic regression parameters. 
  %               You should set p to a vector of 0's and 1's
  %
  
  for i = 1:m
      if sigmoid(X(i,:)*theta) >= 0.5
          p(i) = 1;
      else
          p(i) = 0;
  end
  
  % =========================================================================
  
  end
  ```

- 验证predict.m 函数是否正确

  ```matlab
  % Compute accuracy on our training set
  >> p = predict(theta, X);
  >> fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Logistic Regression...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                            Sigmoid Function |   5 /   5 | Nice work!
  ==                    Logistic Regression Cost |  30 /  30 | Nice work!
  ==                Logistic Regression Gradient |  30 /  30 | Nice work!
  ==                                     Predict |   5 /   5 | Nice work!
  ==        Regularized Logistic Regression Cost |   0 /  15 | 
  ==    Regularized Logistic Regression Gradient |   0 /  15 | 
  ==                                   --------------------------------
  ==                                             |  70 / 100 | 
  == 
  ```



### 2. 正则化逻辑回归  Regularized logistic regression

- 上传训练集

  ```matlab
  >> data = load('ex2data2.txt');
  >> X = data(:, [1, 2]);
  >> y = data(:, 3);
  ```

- 用之前编写好的 plotData.m 可视化训练集

  ```matlab
  >> plotData(X, y);
  
  % Put some labels 
  >> hold on;
  
  % Labels and Legend
  >> xlabel('Microchip Test 1')
  >> ylabel('Microchip Test 2')
  % Specified in plot order
  >> legend('y = 1', 'y = 0')
  >> hold off;
  ```

  ![image-20200611231837962](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex2/image-20200611231837962.png)

- 将训练集 X 中样本的2个特征扩充至28个特征

  这会使得普通梯度下降算法会造成过拟合问题，从而需要应用正则化方法

  ```matlab
  >> X = mapFeature(X(:,1), X(:,2));
  ```

  

#### 2.1 代价函数和梯度  Cost function and gradient

- 打开 costFunctionReg.m 函数，做如下更改

  ```matlab
  function [J, grad] = costFunctionReg(theta, X, y, lambda)
  %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
  %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters. 
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost of a particular choice of theta.
  %               You should set J to the cost.
  %               Compute the partial derivatives and set grad to the partial
  %               derivatives of the cost w.r.t. each parameter in theta
  
  J = (-1 / m) * (y' * log(sigmoid(X*theta)) + (1 - y)' * log(1 - sigmoid(X*theta))) + (lambda / (2*m)) * ((theta(2:end, :))' * (theta(2:end, :)));
  
  grad(1, :) = (1/m) * (sigmoid(X*theta) - y)' * X(:, 1);
  grad(2:end, :) = (1/m) * ((sigmoid(X*theta) - y)' * X(:, 2:end))' + (lambda / m) * theta(2:end, :);
  
  % =============================================================
  
  end
  
  ```

- 在命令行提交

  ```matlab
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==                            Sigmoid Function |   5 /   5 | Nice work!
  ==                    Logistic Regression Cost |  30 /  30 | Nice work!
  ==                Logistic Regression Gradient |  30 /  30 | Nice work!
  ==                                     Predict |   5 /   5 | Nice work!
  ==        Regularized Logistic Regression Cost |  15 /  15 | Nice work!
  ==    Regularized Logistic Regression Gradient |  15 /  15 | Nice work!
  ==                                   --------------------------------
  ==                                             | 100 / 100 | 
  == 
  ```

- 初始化参数

  ```matlab
  initial_theta = zeros(size(X, 2), 1);
  
  lambda = 1;
  ```

- 接下来同之前一样调用fminunc函数求解最优化 theta 即可

