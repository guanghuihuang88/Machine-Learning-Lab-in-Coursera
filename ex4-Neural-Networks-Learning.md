# ex 4: 神经网络  Neural Networks Learning

[toc]

### 1. Neural Networks

- 手写数字识别（1-10分类）



#### 1.1  可视化训练集  Visualizing the data

```matlab
>> load('ex4data1.mat');
>> m = size(X, 1);
>> sel = randperm(size(X, 1));
>> sel = sel(1:100);
>> displayData(X(sel, :));
```

![image-20200616225033527](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex4/image-20200616225033527.png)



#### 1.2  上传训练好的参数 Loading trained parameters

```matlab
>> load('ex4weights.mat');
```



#### 1.3  前向传播算法和代价函数 Feedforward and cost function

![image-20200707173207536](https://hexo.oss-cn-beijing.aliyuncs.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%A4%A7%E4%BD%9C%E4%B8%9ACoursera/ex4/image-20200707173207536.png)

- 打开 nnCostFunction.m 函数，做如下更改

  ```matlab
  function [J grad] = nnCostFunction(nn_params, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     num_labels, ...
                                     X, y, lambda)
  %NNCOSTFUNCTION Implements the neural network cost function for a two layer
  %neural network which performs classification
  %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  %   X, y, lambda) computes the cost and gradient of the neural network. The
  %   parameters for the neural network are "unrolled" into the vector
  %   nn_params and need to be converted back into the weight matrices. 
  % 
  %   The returned parameter grad should be a "unrolled" vector of the
  %   partial derivatives of the neural network.
  %
  
  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));
  %即reshape(nn_params(1:25*401), 25, 401)
  
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));
  %即reshape(nn_params(25*401:end), 10, 26)
  
  % Setup some useful variables
  m = size(X, 1);
           
  % You need to return the following variables correctly 
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: You should complete the code by working through the
  %               following parts.
  %
  % Part 1: Feedforward the neural network and return the cost in the
  %         variable J. After implementing Part 1, you can verify that your
  %         cost function computation is correct by verifying the cost
  %         computed in ex4.m
  %
  % Part 2: Implement the backpropagation algorithm to compute the gradients
  %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  %         Theta2_grad, respectively. After implementing Part 2, you can check
  %         that your implementation is correct by running checkNNGradients
  %
  %         Note: The vector y passed into the function is a vector of labels
  %               containing values from 1..K. You need to map this vector into a 
  %               binary vector of 1's and 0's to be used with the neural network
  %               cost function.
  %
  %         Hint: We recommend implementing backpropagation using a for-loop
  %               over the training examples if you are implementing it for the 
  %               first time.
  %
  % Part 3: Implement regularization with the cost function and gradients.
  %
  %         Hint: You can implement this around the code for
  %               backpropagation. That is, you can compute the gradients for
  %               the regularization separately and then add them to Theta1_grad
  %               and Theta2_grad from Part 2.
  %
  
  X = [ones(m, 1) X];
  hx = sigmoid(Theta2 * [ones(1, m); sigmoid(Theta1 * X')]);
  yk = zeros(m, num_labels);
  for i = 1: m
      yk(i, y(i)) = 1;
  end
  J = (-1/m) * sum(sum(log(hx') .* yk + (ones(m, num_labels) - yk) .* log(ones(m, num_labels) - hx')));
  
  % -------------------------------------------------------------
  
  % =========================================================================
  
  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
  
  
  end
  ```
  
- 检验 nnCostFunction.m 函数正确性，在命令行输入

  ```matlab
  >> input_layer_size  = 400;  % 20x20 Input Images of Digits
  >> hidden_layer_size = 25;   % 25 hidden units
  >> num_labels = 10;          % 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
  
  % Unroll parameters 
  >> nn_params = [Theta1(:) ; Theta2(:)];
  
  % Weight regularization parameter (we set this to 0 here).
  >> lambda = 0;
  
  >> J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
  
  >> fprintf('Cost at parameters (loaded from ex4weights): %f', J);
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
  ==               Feedforward and Cost Function |  30 /  30 | Nice work!
  ==                   Regularized Cost Function |   0 /  15 | 
  ==                            Sigmoid Gradient |   0 /   5 | 
  ==   Neural Network Gradient (Backpropagation) |   0 /  40 | 
  ==                        Regularized Gradient |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  30 / 100 | 
  == 
  ```



#### 1.4  正则化代价函数 Regularized cost function



- 打开 nnCostFunction.m 函数，第二次添加如下代码

  ```matlab
  function [J grad] = nnCostFunction(nn_params, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     num_labels, ...
                                     X, y, lambda)
  %NNCOSTFUNCTION Implements the neural network cost function for a two layer
  %neural network which performs classification
  %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  %   X, y, lambda) computes the cost and gradient of the neural network. The
  %   parameters for the neural network are "unrolled" into the vector
  %   nn_params and need to be converted back into the weight matrices. 
  % 
  %   The returned parameter grad should be a "unrolled" vector of the
  %   partial derivatives of the neural network.
  %
  
  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));
  %即reshape(nn_params(1:25*401), 25, 401)
  
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));
  %即reshape(nn_params(25*401:end), 10, 26)
  
  % Setup some useful variables
  m = size(X, 1);
           
  % You need to return the following variables correctly 
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: You should complete the code by working through the
  %               following parts.
  %
  % Part 1: Feedforward the neural network and return the cost in the
  %         variable J. After implementing Part 1, you can verify that your
  %         cost function computation is correct by verifying the cost
  %         computed in ex4.m
  %
  % Part 2: Implement the backpropagation algorithm to compute the gradients
  %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  %         Theta2_grad, respectively. After implementing Part 2, you can check
  %         that your implementation is correct by running checkNNGradients
  %
  %         Note: The vector y passed into the function is a vector of labels
  %               containing values from 1..K. You need to map this vector into a 
  %               binary vector of 1's and 0's to be used with the neural network
  %               cost function.
  %
  %         Hint: We recommend implementing backpropagation using a for-loop
  %               over the training examples if you are implementing it for the 
  %               first time.
  %
  % Part 3: Implement regularization with the cost function and gradients.
  %
  %         Hint: You can implement this around the code for
  %               backpropagation. That is, you can compute the gradients for
  %               the regularization separately and then add them to Theta1_grad
  %               and Theta2_grad from Part 2.
  %
  
  X = [ones(m, 1) X];
  hx = sigmoid(Theta2 * [ones(1, m); sigmoid(Theta1 * X')]);
  yk = zeros(m, num_labels);
  for i = 1: m
      yk(i, y(i)) = 1;
  end
  J = (-1/m) * sum(sum(log(hx') .* yk + (ones(m, num_labels) - yk) .* log(ones(m, num_labels) - hx')));
  % 第一次submit
  
  Theta1 = Theta1(:, 2:end);
  Theta2 = Theta2(:, 2:end);
  J = J + (lambda / (2*m)) * (sum(sum(Theta1 .* Theta1)) + sum(sum(Theta2 .* Theta2)));
  % 第二次submit
  
  % -------------------------------------------------------------
  
  % =========================================================================
  
  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
  
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
  ==               Feedforward and Cost Function |  30 /  30 | Nice work!
  ==                   Regularized Cost Function |  15 /  15 | Nice work!
  ==                            Sigmoid Gradient |   0 /   5 | 
  ==   Neural Network Gradient (Backpropagation) |   0 /  40 | 
  ==                        Regularized Gradient |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  45 / 100 | 
  == 
  ```



### 2. 后向传播算法 Backpropagation



#### 2.1  S型函数梯度 Sigmoid gradient



- 打开 sigmoidGradient.m 函数，做如下更改

  ```matlab
  function g = sigmoidGradient(z)
  %SIGMOIDGRADIENT returns the gradient of the sigmoid function
  %evaluated at z
  %   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
  %   evaluated at z. This should work regardless if z is a matrix or a
  %   vector. In particular, if z is a vector or matrix, you should return
  %   the gradient for each element.
  
  g = zeros(size(z));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the gradient of the sigmoid function evaluated at
  %               each value of z (z can be a matrix, vector or scalar).
  
  g = sigmoid(z) .* (ones(size(z)) - sigmoid(z));
  
  % =============================================================
  
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
  ==               Feedforward and Cost Function |  30 /  30 | Nice work!
  ==                   Regularized Cost Function |  15 /  15 | Nice work!
  ==                            Sigmoid Gradient |   5 /   5 | Nice work!
  ==   Neural Network Gradient (Backpropagation) |   0 /  40 | 
  ==                        Regularized Gradient |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  50 / 100 | 
  == 
  ```



#### 2.2  随机初始化 Random initialization



- 打开 randInitializeWeights.m 函数，做如下更改

  ```matlab
  function W = randInitializeWeights(L_in, L_out)
  %RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
  %incoming connections and L_out outgoing connections
  %   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
  %   of a layer with L_in incoming connections and L_out outgoing 
  %   connections. 
  %
  %   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
  %   the first column of W handles the "bias" terms
  %
  
  % You need to return the following variables correctly 
  W = zeros(L_out, 1 + L_in);
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: Initialize W randomly so that we break the symmetry while
  %               training the neural network.
  %
  % Note: The first column of W corresponds to the parameters for the bias unit
  %
  
  epsilon_init = 0.12;
  W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
  
  % =========================================================================
  
  end
  ```

- 初始化Theta1、Theta2

  ```matlab
  >> initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
  >> initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
  >> initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
  ```



#### 2.3  后向传播算法 Backpropagation



- 打开 nnCostFunction.m 函数，第三次添加如下代码

  ```matlab
  function [J grad] = nnCostFunction(nn_params, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     num_labels, ...
                                     X, y, lambda)
  %NNCOSTFUNCTION Implements the neural network cost function for a two layer
  %neural network which performs classification
  %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  %   X, y, lambda) computes the cost and gradient of the neural network. The
  %   parameters for the neural network are "unrolled" into the vector
  %   nn_params and need to be converted back into the weight matrices. 
  % 
  %   The returned parameter grad should be a "unrolled" vector of the
  %   partial derivatives of the neural network.
  %
  
  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));
  %即reshape(nn_params(1:25*401), 25, 401)
  
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));
  %即reshape(nn_params(25*401:end), 10, 26)
  
  % Setup some useful variables
  m = size(X, 1);
           
  % You need to return the following variables correctly 
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: You should complete the code by working through the
  %               following parts.
  %
  % Part 1: Feedforward the neural network and return the cost in the
  %         variable J. After implementing Part 1, you can verify that your
  %         cost function computation is correct by verifying the cost
  %         computed in ex4.m
  %
  % Part 2: Implement the backpropagation algorithm to compute the gradients
  %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  %         Theta2_grad, respectively. After implementing Part 2, you can check
  %         that your implementation is correct by running checkNNGradients
  %
  %         Note: The vector y passed into the function is a vector of labels
  %               containing values from 1..K. You need to map this vector into a 
  %               binary vector of 1's and 0's to be used with the neural network
  %               cost function.
  %
  %         Hint: We recommend implementing backpropagation using a for-loop
  %               over the training examples if you are implementing it for the 
  %               first time.
  %
  % Part 3: Implement regularization with the cost function and gradients.
  %
  %         Hint: You can implement this around the code for
  %               backpropagation. That is, you can compute the gradients for
  %               the regularization separately and then add them to Theta1_grad
  %               and Theta2_grad from Part 2.
  %
  
  X = [ones(m, 1) X];
  hx = sigmoid(Theta2 * [ones(1, m); sigmoid(Theta1 * X')]);
  yk = zeros(m, num_labels);
  for i = 1: m
      yk(i, y(i)) = 1;
  end
  J = (-1/m) * sum(sum(log(hx') .* yk + (ones(m, num_labels) - yk) .* log(ones(m, num_labels) - hx')));
  % 第一次submit
  
  J = J + (lambda / (2*m)) * (sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) + sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end))));
  % 第二次submit
  
  z2 = Theta1 * X'; %25*5000 
  a2 = [ones(1, m); sigmoid(z2)];   %26*5000
  a3 = hx;          %10*5000
  delta3 = a3' - yk;  %5000*10
  delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(z2');  %5000*25
  Theta1_grad = (1/m) * delta2' * X;
  Theta2_grad = (1/m) * delta3' * a2';
  % 第三次submit（对第二次做了一定的更改）
  
  % -------------------------------------------------------------
  
  % =========================================================================
  
  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
  
  end
  ```

- 调用 checkNNGradients.m 函数检查结果

  ```matlab
  checkNNGradients;
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
  ==               Feedforward and Cost Function |  30 /  30 | Nice work!
  ==                   Regularized Cost Function |  15 /  15 | Nice work!
  ==                            Sigmoid Gradient |   5 /   5 | Nice work!
  ==   Neural Network Gradient (Backpropagation) |  40 /  40 | Nice work!
  ==                        Regularized Gradient |   0 /  10 | 
  ==                                   --------------------------------
  ==                                             |  90 / 100 | 
  == 
  ```



#### 2.4  正则化神经网络 Regularized neural networks



- 打开 nnCostFunction.m 函数，第四次添加如下代码：

  ```matlab
  function [J grad] = nnCostFunction(nn_params, ...
                                     input_layer_size, ...
                                     hidden_layer_size, ...
                                     num_labels, ...
                                     X, y, lambda)
  %NNCOSTFUNCTION Implements the neural network cost function for a two layer
  %neural network which performs classification
  %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  %   X, y, lambda) computes the cost and gradient of the neural network. The
  %   parameters for the neural network are "unrolled" into the vector
  %   nn_params and need to be converted back into the weight matrices. 
  % 
  %   The returned parameter grad should be a "unrolled" vector of the
  %   partial derivatives of the neural network.
  %
  
  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));
  %即reshape(nn_params(1:25*401), 25, 401)
  
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));
  %即reshape(nn_params(25*401:end), 10, 26)
  
  % Setup some useful variables
  m = size(X, 1);
           
  % You need to return the following variables correctly 
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  % ====================== YOUR CODE HERE ======================
  % Instructions: You should complete the code by working through the
  %               following parts.
  %
  % Part 1: Feedforward the neural network and return the cost in the
  %         variable J. After implementing Part 1, you can verify that your
  %         cost function computation is correct by verifying the cost
  %         computed in ex4.m
  %
  % Part 2: Implement the backpropagation algorithm to compute the gradients
  %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  %         Theta2_grad, respectively. After implementing Part 2, you can check
  %         that your implementation is correct by running checkNNGradients
  %
  %         Note: The vector y passed into the function is a vector of labels
  %               containing values from 1..K. You need to map this vector into a 
  %               binary vector of 1's and 0's to be used with the neural network
  %               cost function.
  %
  %         Hint: We recommend implementing backpropagation using a for-loop
  %               over the training examples if you are implementing it for the 
  %               first time.
  %
  % Part 3: Implement regularization with the cost function and gradients.
  %
  %         Hint: You can implement this around the code for
  %               backpropagation. That is, you can compute the gradients for
  %               the regularization separately and then add them to Theta1_grad
  %               and Theta2_grad from Part 2.
  %
  
  X = [ones(m, 1) X];
  hx = sigmoid(Theta2 * [ones(1, m); sigmoid(Theta1 * X')]);
  yk = zeros(m, num_labels);
  for i = 1: m
      yk(i, y(i)) = 1;
  end
  J = (-1/m) * sum(sum(log(hx') .* yk + (ones(m, num_labels) - yk) .* log(ones(m, num_labels) - hx')));
  % 第一次submit
  
  J = J + (lambda / (2*m)) * (sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) + sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end))));
  % 第二次submit
  
  z2 = Theta1 * X'; %25*5000 
  a2 = [ones(1, m); sigmoid(z2)];   %26*5000
  a3 = hx;          %10*5000
  delta3 = a3' - yk;  %5000*10
  delta2 = (delta3 * Theta2(:, 2:end)) .* sigmoidGradient(z2');  %5000*25
  Theta1_grad = (1/m) * delta2' * X;
  Theta2_grad = (1/m) * delta3' * a2';
  %第三次submit
  
  Theta1_grad = Theta1_grad + [zeros(size(Theta1, 1), 1) (lambda/m) * Theta1(:, 2:end)];
  Theta2_grad = Theta2_grad + [zeros(size(Theta2, 1), 1) (lambda/m) * Theta2(:, 2:end)];
  %第四次submit
  
  % -------------------------------------------------------------
  
  % =========================================================================
  
  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
  
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
  ==               Feedforward and Cost Function |  30 /  30 | Nice work!
  ==                   Regularized Cost Function |  15 /  15 | Nice work!
  ==                            Sigmoid Gradient |   5 /   5 | Nice work!
  ==   Neural Network Gradient (Backpropagation) |  40 /  40 | Nice work!
  ==                        Regularized Gradient |  10 /  10 | Nice work!
  ==                                   --------------------------------
  ==                                             | 100 / 100 | 
  == 
  ```



#### 2.5  利用fmincg训练参数 Learning parameters using fmincg



- 在命令行执行

  ```matlab
  options = optimset('MaxIter', 50);
  lambda = 1;
  
  % Create "short hand" for the cost function to be minimized
  costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
  
  % Now, costFunction is a function that takes in only one argument (the
  % neural network parameters)
  [nn_params, ~] = fmincg(costFunction, initial_nn_params, options);
  % Obtain Theta1 and Theta2 back from nn_params
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
   
  pred = predict(Theta1, Theta2, X);
  fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
  ```

  

