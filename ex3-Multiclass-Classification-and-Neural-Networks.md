## ex 3: 多类别分类和神经网络  Multi-class Classification and Neural Networks

[toc]


### 1. Multi-class Classification

- 手写数字识别（1-10分类）



#### 1.1  可视化训练集  Visualizing the data

```matlab
>> load('ex3data1.mat');
>> m = size(X, 1);
>> rand_indices = randperm(m);
>> sel = X(rand_indices(1:100), :);
>> displayData(sel);
```

![image-20200616225019953](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex3/image-20200616225019953.png)



#### 1.2  逻辑回归 Logistic Regression

- 打开 IrCostFunction.m 函数，做如下更改

  ```matlab
  function [J, grad] = lrCostFunction(theta, X, y, lambda)
  %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
  %regularization
  %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
  %
  % Hint: The computation of the cost function and gradients can be
  %       efficiently vectorized. For example, consider the computation
  %
  %           sigmoid(X * theta)
  %
  %       Each row of the resulting matrix will contain the value of the
  %       prediction for that example. You can make use of this to vectorize
  %       the cost function and gradient computations. 
  %
  % Hint: When computing the gradient of the regularized cost function, 
  %       there're many possible vectorized solutions, but one solution
  %       looks like:
  %           grad = (unregularized gradient for logistic regression)
  %           temp = theta; 
  %           temp(1) = 0;   % because we don't add anything for j = 0  
  %           grad = grad + YOUR_CODE_HERE (using the temp variable)
  %
  
  J = (-1 / m) * (y' * log(sigmoid(X*theta)) + (1 - y)' * log(1 - sigmoid(X*theta))) + (lambda / (2*m)) * ((theta(2:end, :))' * (theta(2:end, :)));
  
  grad(1, :) = (1/m) * (sigmoid(X*theta) - y)' * X(:, 1);
  grad(2:end, :) = (1/m) * ((sigmoid(X*theta) - y)' * X(:, 2:end))' + (lambda / m) * theta(2:end, :);
  
  % =============================================================
  
  grad = grad(:);
  
  end
  ```

- 检验 lrCostFunction.m 函数正确性，在命令行输入

  ```matlab
  >> theta_t = [-2; -1; 1; 2];
  >> X_t = [ones(5,1) reshape(1:15,5,3)/10];
  >> y_t = ([1;0;1;0;1] >= 0.5);
  >> lambda_t = 3;
  
  >> [J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
  
  >> fprintf('Cost: %f | Expected cost: 2.534819\n',J);
  Cost: 2.534819 | Expected cost: 2.534819
  >> fprintf('Gradients:\n'); fprintf('%f\n',grad);
  Gradients:
  0.146561
  -0.548558
  0.724722
  1.398003
  >> fprintf('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003');
  Expected gradients:
   0.146561
   -0.548558
   0.724722
   1.398003
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Multi-class Classification and Neural Networks...
  Login (email address): 
  guanghuihuang88@gmail.com
  Token: 
  ekG9JrKnvxhpCmKJ
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==             Regularized Logistic Regression |  30 /  30 | Nice work!
  ==              One-vs-All Classifier Training |   0 /  20 | 
  ==            One-vs-All Classifier Prediction |   0 /  20 | 
  ==          Neural Network Prediction Function |   0 /  30 | 
  ==                                   --------------------------------
  ==                                             |  30 / 100 | 
  == 
  ```



#### 1.3 一对多分类  One-vs-all classication

- 打开 oneVsAll.m 函数，做如下更改

  ```matlab
  function [all_theta] = oneVsAll(X, y, num_labels, lambda)
  %ONEVSALL trains multiple logistic regression classifiers and returns all
  %the classifiers in a matrix all_theta, where the i-th row of all_theta 
  %corresponds to the classifier for label i
  %   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
  %   logistic regression classifiers and returns each of these classifiers
  %   in a matrix all_theta, where the i-th row of all_theta corresponds 
  %   to the classifier for label i
  
  % Some useful variables
  m = size(X, 1);
  n = size(X, 2);
  
  % You need to return the following variables correctly 
  all_theta = zeros(num_labels, n + 1);
  
  % Add ones to the X data matrix
  X = [ones(m, 1) X];
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: You should complete the following code to train num_labels
  %               logistic regression classifiers with regularization
  %               parameter lambda. 
  %
  % Hint: theta(:) will return a column vector.
  %
  % Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
  %       whether the ground truth is true/false for this class.
  %
  % Note: For this assignment, we recommend using fmincg to optimize the cost
  %       function. It is okay to use a for-loop (for c = 1:num_labels) to
  %       loop over the different classes.
  %
  %       fmincg works similarly to fminunc, but is more efficient when we
  %       are dealing with large number of parameters.
  %
  % Example Code for fmincg:
  %
  %     % Set Initial theta
  %     initial_theta = zeros(n + 1, 1);
  %     
  %     % Set options for fminunc
  %     options = optimset('GradObj', 'on', 'MaxIter', 50);
  % 
  %     % Run fmincg to obtain the optimal theta
  %     % This function will return theta and the cost 
  %     [theta] = ...
  %         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
  %                 initial_theta, options);
  %
  
  initial_theta = zeros(n + 1, 1);
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  for i = 1:num_labels
      [theta, cost] = fmincg(@(t)(lrCostFunction(t, X, (y == i), lambda)), initial_theta, options);
      all_theta(i, :) = theta';
  end
  
  % =========================================================================
  
  
  end
  ```

- 检验 oneVsAll.m 函数正确性，在命令行输入

  ```matlab
  >> num_labels = 10; % 10 labels, from 1 to 10 
  >> lambda = 0.1;
  >> [all_theta] = oneVsAll(X, y, num_labels, lambda);
  
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Multi-class Classification and Neural Networks...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==             Regularized Logistic Regression |  30 /  30 | Nice work!
  ==              One-vs-All Classifier Training |  20 /  20 | Nice work!
  ==            One-vs-All Classifier Prediction |   0 /  20 | 
  ==          Neural Network Prediction Function |   0 /  30 | 
  ==                                   --------------------------------
  ==                                             |  50 / 100 | 
  == 
  ```



#### 1.4 多分类预测  One-vs-all prediction

- 打开 predictOneVsAll.m 函数，做如下更改

  ```matlab
  function p = predictOneVsAll(all_theta, X)
  %PREDICT Predict the label for a trained one-vs-all classifier. The labels 
  %are in the range 1..K, where K = size(all_theta, 1). 
  %  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
  %  for each example in the matrix X. Note that X contains the examples in
  %  rows. all_theta is a matrix where the i-th row is a trained logistic
  %  regression theta vector for the i-th class. You should set p to a vector
  %  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
  %  for 4 examples) 
  
  m = size(X, 1);
  num_labels = size(all_theta, 1);
  
  % You need to return the following variables correctly 
  p = zeros(size(X, 1), 1);
  
  % Add ones to the X data matrix
  X = [ones(m, 1) X];
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Complete the following code to make predictions using
  %               your learned logistic regression parameters (one-vs-all).
  %               You should set p to a vector of predictions (from 1 to
  %               num_labels).
  %
  % Hint: This code can be done all vectorized using the max function.
  %       In particular, the max function can also return the index of the 
  %       max element, for more information see 'help max'. If your examples 
  %       are in rows, then, you can use max(A, [], 2) to obtain the max 
  %       for each row.
  %       
  
  [M, p] = max(sigmoid(X * (all_theta)'), [], 2); 
  
  % =========================================================================
  
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Multi-class Classification and Neural Networks...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==             Regularized Logistic Regression |  30 /  30 | Nice work!
  ==              One-vs-All Classifier Training |  20 /  20 | Nice work!
  ==            One-vs-All Classifier Prediction |  20 /  20 | Nice work!
  ==          Neural Network Prediction Function |   0 /  30 | 
  ==                                   --------------------------------
  ==                                             |  70 / 100 | 
  == 
  ```



### 2. 神经网络  Neural Networks



#### 2.1 前向传播算法  Feedforward propagation and prediction

- 打开 predict.m 函数，做如下更改

  ```matlab
  function p = predict(Theta1, Theta2, X)
  %PREDICT Predict the label of an input given a trained neural network
  %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
  %   trained weights of a neural network (Theta1, Theta2)
  
  % Useful values
  m = size(X, 1);
  num_labels = size(Theta2, 1);
  
  % You need to return the following variables correctly 
  p = zeros(size(X, 1), 1);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Complete the following code to make predictions using
  %               your learned neural network. You should set p to a 
  %               vector containing labels between 1 to num_labels.
  %
  % Hint: The max function might come in useful. In particular, the max
  %       function can also return the index of the max element, for more
  %       information see 'help max'. If your examples are in rows, then, you
  %       can use max(A, [], 2) to obtain the max for each row.
  %
  
  X = [ones(size(X, 1), 1) X];
  
  a = sigmoid(X * Theta1');
  
  a = [ones(size(a, 1), 1) a];
  
  [M, p] = max(sigmoid(a * Theta2'), [], 2); 
  
  % =========================================================================
  
  
  end
  ```

- 在命令行提交

  ```matlab
  >> submit
  == Submitting solutions | Multi-class Classification and Neural Networks...
  Use token from last successful submission (guanghuihuang88@gmail.com)? (Y/n): 
  Y
  == 
  ==                                   Part Name |     Score | Feedback
  ==                                   --------- |     ----- | --------
  ==             Regularized Logistic Regression |  30 /  30 | Nice work!
  ==              One-vs-All Classifier Training |  20 /  20 | Nice work!
  ==            One-vs-All Classifier Prediction |  20 /  20 | Nice work!
  ==          Neural Network Prediction Function |  30 /  30 | Nice work!
  ==                                   --------------------------------
  ==                                             | 100 / 100 | 
  == 
  ```









