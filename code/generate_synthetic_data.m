%clear all; close all; clc;
function [data]  = generate_synthetic_data()
ext_feat = 5;
n_test = 2000;
n_train = 10000;
n_examples = n_test+n_train;
dist = 3.5;

mu = [0 0];
Sigma = [1 0; 0 0.3]; R = chol(Sigma);
x = (repmat(mu,n_examples,1) + randn(n_examples,2)*R)';
x(2,1:n_examples/2) = x(2,1:n_examples/2)+dist;
x(2,:) = x(2,:)- dist/2;
y = x(2,:)>0;
margin = abs(x(2,:));

angle = 45 / (2*pi);
x(1:2,:) = [cos(angle),sin(angle); -sin(angle),cos(angle)] * x(1:2,:);
figure; hold on
plot(x(1,y==1),x(2,y==1),'o');
plot(x(1,y==0),x(2,y==0),'*r');

[~,idx]=sort(margin(1:n_train));


% figure; hold on
% for ii=n_examples-1:-1:1,
%     plot(xordered(1,ii),xordered(2,ii),'o');
%     pause(0.000001)
% end

%x(3:ext_feat,:) = randn(ext_feat, n_examples);

xordered = x(:,idx);
yordered = y(idx);
data = cell(1,2);

data{1}.train.x = xordered;
data{1}.train.y = yordered;
data{2}.train.x = xordered(:,end:-1:1);
data{2}.train.y = yordered(end:-1:1);
data{1}.test.x = x(:,n_train+1:end);
data{1}.test.y = y(n_train+1:end);
data{2}.test.x = x(:,n_train+1:end);
data{2}.test.y = y(n_train+1:end);
end


