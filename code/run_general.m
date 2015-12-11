

% get data
% predict - input K x, output K y/p
% draw task - input get margins return task
% update - update if need - input task, return

clear all; close all; clc;

%% load data

my_dir = '/Users/haimcohen/Documents/privateDocs/ThesisData/usps';
load (sprintf('%s/%s',my_dir,'usps_data_1vs_rest.mat'));
%load (sprintf('%s/%s',my_dir,'usps_data_pairs.mat'));
digitVec = data.test{1}.vectors(1:end-1,20);
diglabel = data.test{1}.labels(20);
digitImage = reshape(digitVec,[16,16]);
%figure; imshow(digitImage)
num_tasks = length(data.test);

% data = generate_synthetic_data();
% num_tasks = length(data{1}.test);

n_trials = 1;
b_all = 10.^(-7:5);


data1 = data;
clear data;
data = cell(1,num_tasks);

for ii=1:num_tasks
    data{ii}.test.y = data1.test{ii}.labels;
    data{ii}.train.y = data1.train{ii}.labels;
    data{ii}.test.x = data1.test{1}.vectors;
    data{ii}.train.x = data1.train{1}.vectors;
end

maxNorm = 0;
for ii = 1:size(data{1}.train.x,2)
    if(norm(data{1}.train.x(:,ii))>maxNorm)
       maxNorm = norm(data{1}.train.x(:,ii)); 
    end
end

for ii=1:num_tasks
    data{ii}.test.x = data{1}.test.x/maxNorm;
    data{ii}.train.x = data{1}.train.x/maxNorm;
end
%clear data1;



%% run SHAMPO

%number of examples (the minimal)
n = size(data{1}.train.x,2);
for jj=1:num_tasks
    n = min([n,size(data{jj}.train.x, 2)]);
end

params.n_ex = n;





% examples permutations
pt = randperm(n);

all_res = cell(n_trials,length(b_all));
queried = zeros(num_tasks,length(b_all));
train_err_total = zeros(num_tasks,length(b_all));
train_err_queried = zeros(num_tasks,length(b_all));
test_res = zeros(num_tasks, length(b_all));



for ii=1:n_trials
    % donpe = randperm(length(inds_mistakes_easy));
    
    for b_ind = 1:length(b_all)
        fprintf('trial %d.   b index %d  \n',ii,b_ind);
        params.b = b_all(b_ind);
        % training
        res_run = shampo( data, params );
        
        % test
        w = res_run.w;
        for jj=1:num_tasks,
            % task,b value,perceptron method
            test_res(jj, b_ind) = mean( ((w{jj}' * data{jj}.test.x) .* data{jj}.test.y) <=0);
        end
        
    all_res{ii,b_ind}.test_res = test_res;
    all_res{ii,b_ind}.train_error = res_run.train_error;
    all_res{ii,b_ind}.train_err_total = res_run.train_err_total;
    all_res{ii,b_ind}.train_err_queried = res_run.train_err_queried;
    all_res{ii,b_ind}.queried = res_run.queried;
    all_res{ii,b_ind}.margin = res_run.margin;
    all_res{ii,b_ind}.b = params.b;
       
    end
end

% predict - input K x, output K y/p