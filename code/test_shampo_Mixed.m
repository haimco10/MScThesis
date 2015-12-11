% Shampo tests

clear all; close all; clc;
%dbstop if all error

%% in order to allow parfor

% pools = matlabpool('size');
% if (pools~=0)
%     matlabpool('close');
% end
% matlabpool('open');

% cpus = feature('numCores');
% if pools ~= (cpus - 1)
%     if pools > 0
%         matlabpool('close');
%     end
%     matlabpool('open', cpus - 1);
% end


%%

%dataset = 'usps_pairs';
%dataset = 'mnist_pairs';
%dataset = 'vj_pairs';
dataset = 'usps_onerest';
%dataset = 'mnist_onerest';
%dataset = 'NLP';
%dataset = 'vj_onerest';
%dataset = 'Mixed';

n_hard_from = 0.7;
n_easy_to = 0.3;

n_easy_to = 6;
n_hard_to = 12;

min_examples = 1e2;%1e6;
b_all = 10.^(-7:5);
n_trials = 1;
c = 0.001;
max_n_tasks = 6;
th = 0.05:0.05:0.2;

%data = get_data(dataset);
my_dir = '/Users/haimcohen/Documents/privateDocs/ThesisData/usps';
load (sprintf('%s/%s',my_dir,'usps_data_pairs.mat'));
if strcmp(dataset,'NLP')
    data{1}.test = [];
end
% if (isempty(data{1}.test))data{task_ind}.train.y(jj)
%     fprintf('need number of folds for cross validation,\n there is no test set\n');
% end

%% train models to determine the easy and hard
% run simple perceptron on all tasks to determine the hardness
n_all_tasks = length(data);
num_mistakes = zeros(n_all_tasks,1);

n_ex = zeros(1,n_all_tasks);
mind = inf;
maxd=0;

for ii=1:n_all_tasks
    [a,b] = size(data{ii}.train.x);
    n_ex(ii) = n_ex(ii)+b;
    if (a<mind) 
        mind=a;
    end
        if(a>maxd) 
            maxd=a; 
        end
end


for ii=1:n_all_tasks,
    %if size(data{ii}.train.x,2)>4000
    x = data{ii}.train.x;
    y = data{ii}.train.y;
    w = zeros(size(x, 1), 1);
    for jj=1:size(x,2),
        if (w' * x(:, jj) * y(jj)) <= 0,
            num_mistakes(ii) =       num_mistakes(ii) + 1;
            w = w + x(:, jj) * y(jj);
        end
    end
    %end
end


% split to hard and easy examples.

[~, inds_mistakes] = sort(num_mistakes'./n_ex);
% n_easy_to = ceil(n_easy_to*n_all_tasks);
% n_hard_from = max([1,floor(n_hard_from*n_all_tasks)]);
% inds_mistakes_easy = inds_mistakes(1:n_easy_to);
% inds_mistakes_hard = inds_mistakes(n_hard_from:n_all_tasks);
inds_mistakes( n_ex<min_examples) = []; 
inds_mistakes_easy = inds_mistakes(1:n_easy_to);
inds_mistakes_hard = inds_mistakes(n_easy_to+1:n_hard_to);

n_hard = 4;
n_easy = 2;

% % shuffle the examples
% pe = randperm(length(inds_mistakes_easy));
% ph = randperm(length(inds_mistakes_hard));

% for num_tasks = 2:max_n_tasks
%     for n_easy = 1:(num_tasks-1)
%         n_hard = num_tasks-n_easy;
%         pe = randperm(length(inds_mistakes_easy));
%         ph = randperm(length(inds_mistakes_hard));
%         pt = [inds_mistakes_easy(pe(1:n_easy));inds_mistakes_hard(ph(1:n_hard))];

        % plain
        run_shampo_diag_new(n_hard,n_easy,n_hard_from,n_easy_to,min_examples,...
            b_all,0,n_trials,dataset,'CB','plain',0,data,inds_mistakes_easy,inds_mistakes_hard);
        % agressive 1
        run_shampo_diag_new(n_hard,n_easy,n_hard_from,n_easy_to,min_examples,...
            b_all,0.05,n_trials,dataset,'CB','aggressive',0,data,inds_mistakes_easy,inds_mistakes_hard);
        % agressive 2
        run_shampo_diag_new(n_hard,n_easy,n_hard_from,n_easy_to,min_examples,...
            b_all,0.1,n_trials,dataset,'CB','aggressive',0,data,inds_mistakes_easy,inds_mistakes_hard);
        % agressive 3
        run_shampo_diag_new(n_hard,n_easy,n_hard_from,n_easy_to,min_examples,...
            b_all,0.15,n_trials,dataset,'CB','aggressive',0,data,inds_mistakes_easy,inds_mistakes_hard);
        % agressive 4
        run_shampo_diag_new(n_hard,n_easy,n_hard_from,n_easy_to,min_examples,...
            b_all,0.2,n_trials,dataset,'CB','aggressive',0,data,inds_mistakes_easy,inds_mistakes_hard);
        % Second order
        run_shampo_diag_new(n_hard,n_easy,n_hard_from,n_easy_to,min_examples,...
            b_all,0,n_trials,dataset,'SO','aggressive',0,data,inds_mistakes_easy,inds_mistakes_hard);
        %AROW
        run_shampo_diag_new(n_hard,n_easy,n_hard_from,n_easy_to,min_examples,...
            b_all,0,n_trials,dataset,'AROW','aggressive',c,data,inds_mistakes_easy,inds_mistakes_hard);
        %aggressiv with b
        run_shampo_diag_new(n_hard,n_easy,n_hard_from,n_easy_to,min_examples,...
            b_all,0,n_trials,dataset,'CB','b',c,data,inds_mistakes_easy,inds_mistakes_hard);
%     end
% end

matlabpool('close');




