function [ res ] = shampo( data,params )
%SHAMPO
%   Detailed explanation goes here
%


b = params.b;
n_tasks = length(data);
n_ex = params.n_ex; %number of examples (the minimal)

queried = zeros(n_ex, 1);
mar  = zeros(n_tasks, 1);
train_error = false(n_tasks,n_ex);
train_err_queried  = false(n_ex, 1);
train_err_total = zeros(n_tasks, 1);
margin = zeros(n_tasks, n_ex);
lambda = 0;


w = cell( n_tasks, 1);
for ii=1:n_tasks,
    w{ii} = zeros( size(data{ii}.train.x, 1), 1);
end

for jj=1:n_ex,
    for ii=1:n_tasks,
        mar(ii) = w{ii}' * data{ii}.train.x(:, jj);
        if (mar(ii)*data{ii}.train.y(jj)<=0)
            train_err_total(ii) = train_err_total(ii)+1;
            train_error(ii,jj) = 1;
        end
    end;
    margin(:,jj) = mar;
    min_mar = min(abs(mar));
    prob = 1./ (abs(mar) - min_mar + b);
    prob = prob / sum(prob);
    task_ind = sum( (rand(1) >= cumsum(prob))) + 1; 
    queried(jj) = task_ind;
    if (mar(task_ind) * data{task_ind}.train.y(jj) <= lambda),
        if (mar(task_ind) * data{task_ind}.train.y(jj) <= 0)
            train_err_queried (jj) = 1;
        end
        w{task_ind} = w{task_ind} + data{task_ind}.train.y(jj) * data{task_ind}.train.x(:,jj);
    end
end



clear mar
clear prob
res.w = w;
res.queried = queried;
res.margin = margin;
res.train_err_queried = train_err_queried ;
res.train_err_total = train_err_total ;
res.train_error = train_error;

end

% task_ind = sum( (rand(1) >= cumsum(prob))) + 1; --- this seems to be the most efficient way to sample
% http://stackoverflow.com/questions/2977497/weighted-random-numbers-in-matlab


