% Jonas Tjomsland - jt2718@ic.ac.uk - CID: 01570830 - MSc HBR

load('parameters')
load('test_data')
input = test_data(:,2:end);
correct_labels = test_data(:,1);

predicted_labels = ClassifyX(input, parameters);

diff = predicted_labels - predicted_labels;
idx=diff==0;
accuracy = sum(idx(:))/length(correct_labels)



function predicted_labels = ClassifyX(input, parameters)
%% All implementation should be inside the function.

% Normalize data:
% Calculating mean and std for all features:
means = [];
stds = [];
for feature = 1:size(input,2)
    means = [means, mean(input(:,feature))];
    stds = [stds, std(input(:,feature))];
end
% Normalizing every features data:
for i = 1:size(input,2)
    input(:,i) = ((input(:,i) - transpose(means(i)))./ stds(i));
end

scores = forward_backward(input, parameters);

predicted_labels = [];
for i = 1:size(input,1)
    [value, index] = max(scores(i,:));
    predicted_labels = [predicted_labels; index];
end


function scores = forward_backward(X,param)
    
    % Implement forward propagation:
    % First layer:
    [z1, cache_first_layer] = linear_forward(X, param{1}, param{2});
    
    % ReLU function
    [a1, cache_first_relu] = relu_forward(z1);

    
    % Second layer
    [z2, cache_second_layer] = linear_forward(a1, param{3}, param{4});
    
    % ReLU function
    [a2, cache_second_relu] = relu_forward(z2);    
    
    % Second layer
    [z3, cache_third_layer] = linear_forward(a2, param{5}, param{6});    
    
    % ReLU function
    [a3, cache_third_relu] = relu_forward(z3); 
    
    % Second layer
    [z4, cache_fourth_layer] = linear_forward(a3, param{7}, param{8});
    
    scores = z4;
   
end

% Function that computes forward pass for one linear layer. Takes the input
% X, a matrix of weights W and a bias term B. Returns the output z and a
% cache containing X, W & B.
function [z, cache] = linear_forward(X, W, B)
    
    z = X*W + B;
    cache = {X, W, B};   

end

% Function that takes input X of any shape and returns an output of the
% same shape, passed through the relu function, as well as a cache.
function [a, cache] = relu_forward(X)
    
    a = max(0,X); 
    cache = X;
    
end

end