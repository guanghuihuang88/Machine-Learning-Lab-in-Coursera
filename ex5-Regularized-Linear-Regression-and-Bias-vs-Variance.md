# ex5:  正则化线性回归和偏差vs.方差 Regularized Linear Regression and Bias vs. Variance

[toc]

### 1. Regularized Linear Regression



#### 1.1  可视化训练集  Visualizing the data

```matlab
% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
>> load ('ex5data1.mat');
% m = Number of examples
>> m = size(X, 1);

% Plot training data
>> figure;
>> plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
>> xlabel('Change in water level (x)');
>> ylabel('Water flowing out of the dam (y)');
```

![image-20200713200309496](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex5/image-20200713200309496.png)



#### 1.2  正则化线性回归代价函数 Regularized linear regression cost function



- 打开 linearRegCostFunction.m 函数，做如下更改

  ```matlab
  function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
  %regression with multiple variables
  %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
  %   cost of using theta as the parameter for linear regression to fit the 
  %   data points in X and y. Returns the cost in J and the gradient in grad
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost and gradient of regularized linear 
  %               regression for a particular choice of theta.
  %
  %               You should set J to the cost and grad to the gradient.
  %
  
  J = (1 / (2*m)) * sum((X * theta - y) .^ 2) + (lambda / (2*m)) * sum(theta(2: end) .^ 2);
      
  % =========================================================================
  
  grad = grad(:);
  
  end
  
  ```
  
- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Neural Networks Learning...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  == Regularized Linear Regression Cost Function |  25 /  25 | Nice work!
  ==      Regularized Linear Regression Gradient |   0 /  25 | 
  ==                              Learning Curve |   0 /  20 | 
  ==                  Polynomial Feature Mapping |   0 /  10 | 
  ==                            Validation Curve |   0 /  20 | 
  ==                                   --------------------------------
  ==                                             |  25 / 100 | 
  
  ```

  

#### 1.3  正则化线性回归梯度 Regularized linear regression gradient



- 打开 linearRegCostFunction.m 函数，添加如下代码

  ```matlab
  function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
  %regression with multiple variables
  %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
  %   cost of using theta as the parameter for linear regression to fit the 
  %   data points in X and y. Returns the cost in J and the gradient in grad
  
  % Initialize some useful values
  m = length(y); % number of training examples
  
  % You need to return the following variables correctly 
  J = 0;
  grad = zeros(size(theta));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost and gradient of regularized linear 
  %               regression for a particular choice of theta.
  %
  %               You should set J to the cost and grad to the gradient.
  %
  
  J = (1 / (2*m)) * sum((X * theta - y) .^ 2) + (lambda / (2*m)) * sum(theta(2: end) .^ 2);
  %第一次submit
  
  grad(1) = (1/m) * (X(:, 1)' * (X * theta - y));
  grad(2: end) = (1/m) * (X(:, 2: end)' * (X * theta - y)) + (lambda/m) * theta(2: end);
  %第二次submit
  
  % =========================================================================
  
  grad = grad(:);
  
  end
  
  ```
  
- 验证函数正确性

  ```matlab
  [J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
  fprintf('Gradient at theta = [1 ; 1]:  [%f; %f] \n',grad(1), grad(2));
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Regularized Linear Regression and Bias/Variance...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  == Regularized Linear Regression Cost Function |  25 /  25 | Nice work!
  ==      Regularized Linear Regression Gradient |  25 /  25 | Nice work!
  ==                              Learning Curve |   0 /  20 | 
  ==                  Polynomial Feature Mapping |   0 /  10 | 
  ==                            Validation Curve |   0 /  20 | 
  ==                                   --------------------------------
  ==                                             |  50 / 100 | 
  == 
  ```




#### 1.4  拟合线性回归 Fitting linear regression



- 打开 trainLinearReg.m 函数，代码已帮你写出，该函数利用 fmincg 用来对给定数据集训练 theta 参数

  ```matlab
  lambda = 0;
  [theta] = trainLinearReg([ones(m, 1) X], y, lambda);
  ```

- 对拟合好的参数作图

  ```matlab
  %  Plot fit over the data
  figure;
  plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
  xlabel('Change in water level (x)');
  ylabel('Water flowing out of the dam (y)');
  hold on;
  plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
  hold off;
  ```

  ![image-20200714143409880](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex5/image-20200714143409880.png)



### 2. 偏差/方差 Bias-variance



#### 2.1  学习曲线 Learning curves



- 打开 learningCurve.m 函数，做如下更改

  ```matlab
  function [error_train, error_val] = ...
      learningCurve(X, y, Xval, yval, lambda)
  %LEARNINGCURVE Generates the train and cross validation set errors needed 
  %to plot a learning curve
  %   [error_train, error_val] = ...
  %       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
  %       cross validation set errors for a learning curve. In particular, 
  %       it returns two vectors of the same length - error_train and 
  %       error_val. Then, error_train(i) contains the training error for
  %       i examples (and similarly for error_val(i)).
  %
  %   In this function, you will compute the train and test errors for
  %   dataset sizes from 1 up to m. In practice, when working with larger
  %   datasets, you might want to do this in larger intervals.
  %
  
  % Number of training examples
  m = size(X, 1);
  
  % You need to return these values correctly
  error_train = zeros(m, 1);
  error_val   = zeros(m, 1);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Fill in this function to return training errors in 
  %               error_train and the cross validation errors in error_val. 
  %               i.e., error_train(i) and 
  %               error_val(i) should give you the errors
  %               obtained after training on i examples.
  %
  % Note: You should evaluate the training error on the first i training
  %       examples (i.e., X(1:i, :) and y(1:i)).
  %
  %       For the cross-validation error, you should instead evaluate on
  %       the _entire_ cross validation set (Xval and yval).
  %
  % Note: If you are using your cost function (linearRegCostFunction)
  %       to compute the training and cross validation error, you should 
  %       call the function with the lambda argument set to 0.  
  %       Do note that you will still need to use lambda when running
  %       the training to obtain the theta parameters.
  %
  % Hint: You can loop over the examples with the following:
  %
  %       for i = 1:m
  %           % Compute train/cross validation errors using training examples 
  %           % X(1:i, :) and y(1:i), storing the result in 
  %           % error_train(i) and error_val(i)
  %           ....
  %           
  %       end
  %
  
  % ---------------------- Sample Solution ----------------------
  
  for i = 1:m
      theta = trainLinearReg(X(1:i, :), y(1:i), lambda);
      error_train(i) = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
      error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
  end
  
  % -------------------------------------------------------------
  
  % =========================================================================
  
  end
  ```
  
- 计算学习曲线、并可视化

  ```matlab
  lambda = 0;
  [error_train, error_val] = learningCurve([ones(m, 1) X], y, [ones(size(Xval, 1), 1) Xval], yval, lambda);
  
  plot(1:m, error_train, 1:m, error_val);
  title('Learning curve for linear regression')
  legend('Train', 'Cross Validation')
  xlabel('Number of training examples')
  ylabel('Error')
  axis([0 13 0 150])
  
  fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
  for i = 1:m
      fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
  end
  ```
  
  ![image-20200714143705695](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex5/image-20200714143705695.png)
  
- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Regularized Linear Regression and Bias/Variance...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  == Regularized Linear Regression Cost Function |  25 /  25 | Nice work!
  ==      Regularized Linear Regression Gradient |  25 /  25 | Nice work!
  ==                              Learning Curve |  20 /  20 | Nice work!
  ==                  Polynomial Feature Mapping |   0 /  10 | 
  ==                            Validation Curve |   0 /  20 | 
  ==                                   --------------------------------
  ==                                             |  70 / 100 | 
  == 
  ```



### 3.  多项式回归 Polynomial regression



- 打开 polyFeatures.m 函数，做如下更改

  ```matlab
  function [X_poly] = polyFeatures(X, p)
  %POLYFEATURES Maps X (1D vector) into the p-th power
  %   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
  %   maps each example into its polynomial features where
  %   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
  %
  
  
  % You need to return the following variables correctly.
  X_poly = zeros(numel(X), p);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Given a vector X, return a matrix X_poly where the p-th 
  %               column of X contains the values of X to the p-th power.
  %
  % 
  
  for i = 1 : p
      X_poly(:, i) = X .^ i;
  end
  
  % =========================================================================
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Regularized Linear Regression and Bias/Variance...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  ==
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  == Regularized Linear Regression Cost Function |  25 /  25 | Nice work!
  ==      Regularized Linear Regression Gradient |  25 /  25 | Nice work!
  ==                              Learning Curve |  20 /  20 | Nice work!
  ==                  Polynomial Feature Mapping |  10 /  10 | Nice work!
  ==                            Validation Curve |   0 /  20 | 
  ==                                   --------------------------------
  ==                                             |  80 / 100 | 
  == 
  ```



#### 3.1  训练多项式回归 Learning Polynomial Regression



- 注意，上面准备好的多项式训练集的数值会很大（比如，p = 5，x = 40， x^5 很大），需要先用函数featureNormalize.m 正则化多项式特征 X_poly

  ```matlab
  p = 8;
  
  % Map X onto Polynomial Features and Normalize
  X_poly = polyFeatures(X, p);
  [X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
  X_poly = [ones(m, 1), X_poly];                   % Add Ones
  
  ```

- 同时用相同的正则化处理交叉验证集 X_poly_val 和测试集 X_poly_test

  ```matlab
  % % Map X_poly_test and normalize (using mu and sigma)
  X_poly_test = polyFeatures(Xtest, p);
  X_poly_test = X_poly_test-mu; % uses implicit expansion instead of bsxfun
  X_poly_test = X_poly_test./sigma; % uses implicit expansion instead of bsxfun
  X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones
  
  % Map X_poly_val and normalize (using mu and sigma)
  X_poly_val = polyFeatures(Xval, p);
  X_poly_val = X_poly_val-mu; % uses implicit expansion instead of bsxfun
  X_poly_val = X_poly_val./sigma; % uses implicit expansion instead of bsxfun
  X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones
  
  fprintf('Normalized Training Example 1:\n');
  fprintf('  %f  \n', X_poly(1, :));
  ```

- 利用正则化的训练集训练好参数 theta

  ```matlab
  lambda = 0;
  [theta] = trainLinearReg(X_poly, y, lambda);
  ```

  

#### 3.2  利用交叉验证集选择λ Selecting lambda using a cross validation set



- 打开 validationCurve.m 函数，做如下更改

  ```matlab
  function [lambda_vec, error_train, error_val] = ...
      validationCurve(X, y, Xval, yval)
  %VALIDATIONCURVE Generate the train and validation errors needed to
  %plot a validation curve that we can use to select lambda
  %   [lambda_vec, error_train, error_val] = ...
  %       VALIDATIONCURVE(X, y, Xval, yval) returns the train
  %       and validation errors (in error_train, error_val)
  %       for different values of lambda. You are given the training set (X,
  %       y) and validation set (Xval, yval).
  %
  
  % Selected values of lambda (you should not change this)
  lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
  
  % You need to return these variables correctly.
  error_train = zeros(length(lambda_vec), 1);
  error_val = zeros(length(lambda_vec), 1);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Fill in this function to return training errors in 
  %               error_train and the validation errors in error_val. The 
  %               vector lambda_vec contains the different lambda parameters 
  %               to use for each calculation of the errors, i.e, 
  %               error_train(i), and error_val(i) should give 
  %               you the errors obtained after training with 
  %               lambda = lambda_vec(i)
  %
  % Note: You can loop over lambda_vec with the following:
  %
  %       for i = 1:length(lambda_vec)
  %           lambda = lambda_vec(i);
  %           % Compute train / val errors when training linear 
  %           % regression with regularization parameter lambda
  %           % You should store the result in error_train(i)
  %           % and error_val(i)
  %           ....
  %           
  %       end
  %
  %
  
  for i = 1:length(lambda_vec)
      theta = trainLinearReg(X, y, lambda_vec(i));
      error_train(i) = linearRegCostFunction(X, y, theta, 0);
      error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
  end
  
  % =========================================================================
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Regularized Linear Regression and Bias/Variance...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  == Regularized Linear Regression Cost Function |  25 /  25 | Nice work!
  ==      Regularized Linear Regression Gradient |  25 /  25 | Nice work!
  ==                              Learning Curve |  20 /  20 | Nice work!
  ==                  Polynomial Feature Mapping |  10 /  10 | Nice work!
  ==                            Validation Curve |  20 /  20 | Nice work!
  ==                                   --------------------------------
  ==                                             | 100 / 100 | 
  == 
  ```



