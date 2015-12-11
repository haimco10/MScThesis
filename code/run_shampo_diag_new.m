function [ all_res] = run_shampo_diag_new(n_hard,n_easy,n_hard_from,n_easy_to,min_examples_all,...
    b_all,aggressive_th,n_trials,dataset,alg,perceptron_update,c,data,inds_mistakes_easy,inds_mistakes_hard)
%RUN_SHAMPO gets parameter and run SHAMPO algorithm on the desired data set.
%   Detailed explanation goes here
%   Haim Cohen, February 2014, Technion

%dbstop if all error

%Inputs:
% n_hard = 5;
% n_easy = 4;
% n_hard_from = 0.6;
% n_easy_to = 0.4;
% min_examples = 5000;
num_folds = 4;
%save_every = n_trials;
% b_all = 10.^(-7:5);%[1e-5, 1e-2, 1e3];
% aggressive_th = 0.2;
% perceptron_update= 'plain';%'aggressive'; % 
% dataset = 'usps_onerest';%'usps_pairs';%'20_news_3';
% alg = 'CB';
% n_trials = 5;

num_tasks = n_hard + n_easy;

time_stamp = strrep(cat(1,num2str(clock)),' ','');
params.update = perceptron_update;
params.alg = alg;
params.n_hard_from = n_hard_from;
params.n_easy_to = n_easy_to;
params.b_all = b_all;
params.aggressive_th = aggressive_th;
params.num_tasks = num_tasks;
params.dataset = dataset;
params.n_trials = n_trials;
params.n_hard = n_hard;
params.n_easy = n_easy;
params.c = c;

fn = sprintf('res_all_%s_%s_%s_%d_%d_%1.2f_%1.2f_%d_%d_%s.mat',dataset,alg, ...
    perceptron_update, n_hard,n_easy,aggressive_th,c,min_examples_all,n_trials,time_stamp);

% data = get_data(dataset);
% if strcmp(dataset,'NLP')
%     data{1}.test = [];
% end
% if (isempty(data{1}.test))&&(num_folds==0)
%     fprintf('need number of folds for cross validation,\n there is no test set\n');  
% end
% 
% %% train models to determine the easy and hard
% % run simple perceptron on all tasks to determine the hardness
n_all_tasks = length(data);
% num_mistakes = zeros(n_all_tasks,1);
% for ii=1:n_all_tasks,
%   x = data{ii}.train.x;
%   y = data{ii}.train.y;
%   w = zeros(size(x, 1), 1);
%   for jj=1:size(x,2),
%     if (w' * x(:, jj) * y(jj)) <= 0,
%       num_mistakes(ii) =       num_mistakes(ii) + 1;
%       w = w + x(:, jj) * y(jj);
%     end
%   end
% end
% 
% % split to hard and easy examples.
% 
% [~, inds_mistakes] = sort(num_mistakes);
% n_easy_to = ceil(n_easy_to*n_all_tasks);
% n_hard_from = max([1,floor(n_hard_from*n_all_tasks)]);
% inds_mistakes_easy = inds_mistakes(1:n_easy_to);
% inds_mistakes_hard = inds_mistakes(n_hard_from:n_all_tasks);
% 
% 
% %%
all_res = cell(n_trials,1);
queried = zeros(num_tasks,length(b_all));
train_err_total = zeros(num_tasks,length(b_all));
train_err_queried = zeros(num_tasks,length(b_all));

for ii=1:n_trials
    
    % % shuffle the examples
    pe = randperm(length(inds_mistakes_easy));
    ph = randperm(length(inds_mistakes_hard));
    pt = [inds_mistakes_easy(pe(1:n_easy)) inds_mistakes_hard(ph(1:n_hard))];
%     params.easy_tasks = pe(1:n_easy);
%     params.hard_tasks = ph(1:n_hard);
    %
    n = size(data{pt(1)}.train.x, 2); %number of examples (the minimal)
    for jj=1:num_tasks
        n = min([n,size(data{pt(jj)}.train.x, 2)]);
    end
    
    min_examples = min(n,min_examples_all);
    params.n_ex = min_examples;
    data1 = cell(1,num_tasks);
    
    % need cross validation:
    %-----------------------------------
    if (isempty(data{1}.test))
        if strcmp(alg,'CB')||strcmp(alg,'AROW')   % for average perceptron
            test_res = zeros( num_folds, num_tasks, length(b_all),2);
        else
            test_res = zeros( num_folds, num_tasks, length(b_all));
        end
        step = floor(min_examples / num_folds);
        for kk=1:num_folds
            inds = (1:step) + (kk-1)*step;
            for jj=1:num_tasks,
                p = randperm(min_examples);
                data1{jj}.train.x = data{pt(jj)}.train.x(:,p);
                data1{jj}.train.y = data{pt(jj)}.train.y(p);
                data1{jj}.test.x = data1{jj}.train.x(:,inds);
                data1{jj}.test.y = data1{jj}.train.y(inds);
                data1{jj}.train.x(:, inds) = [];
                data1{jj}.train.y(inds) = [];
            end
            for b_ind = 1:length(b_all)
                fprintf('trial %d.   fold %d.   b index %d  \n',ii,kk,b_ind);
                params.b = b_all(b_ind);
                % train
                if strcmp(alg,'CB')||strcmp(alg,'AROW') 
                res_run = shampo( data1, params );
                w = res_run.w;
                v = res_run.v;
                % test
                for jj=1:num_tasks,
                    % fold,task,b value,perceptron method
                    test_res(kk, jj, b_ind, 1) = mean( ((w{jj}' * data1{jj}.test.x) .* data1{jj}.test.y) <=0);
                    test_res(kk, jj, b_ind, 2) = mean( ((v{jj}' * data1{jj}.test.x) .* data1{jj}.test.y) <=0);
                end
                elseif (strcmp(alg,'SO'))
                    res_run = second_order_shampo_diag( data1, params );
                    w = res_run.w;
                    invA = res_run.invA;
                    for jj=1:num_tasks
                        % task,b value,perceptron method
                        test_res(kk,jj, b_ind) =  mean(((data1{jj}.test.x)'*((invA{jj}).*w{jj}).*data1{jj}.test.y') <=0);
                    end
%                 elseif (strcmp(alg,'AROW'))
%                     res_run = second_order_shampo_diag( data1, params );
%                     w = res_run.w;
%                     %invA = res_run.invA;
%                     for jj=1:num_tasks
%                         test_res(kk,jj, b_ind) =  mean(((w{jj}' * data1{jj}.test.x) .* data1{jj}.test.y) <=0);
%                     end
                end
                queried(:,b_ind) = res_run.queried;
                train_err_total(:,b_ind) = train_err_total(:,b_ind)+(1/num_folds)*res_run.train_err_total;
                train_err_queried(:,b_ind) = train_err_queried(:,b_ind)+(1/num_folds)*res_run.train_err_queried;
            end
        end 
    else
        % has a test set:
        %---------------------------------------------------
        if strcmp(alg,'CB')||strcmp(alg,'AROW')    % for average perceptron
            test_res = zeros(num_tasks, length(b_all),2);
        else
            test_res = zeros(num_tasks, length(b_all));
        end
        for jj=1:num_tasks
            p = randperm(min_examples);
            data1{jj}.train.x = normc(data{pt(jj)}.train.x(:,p));
            data1{jj}.train.y = data{pt(jj)}.train.y(p);
            data1{jj}.test.x = normc(data{pt(jj)}.test.x);
            data1{jj}.test.y = data{pt(jj)}.test.y;
        end
        for b_ind = 1:length(b_all)
            params.b = b_all(b_ind);
            fprintf('trial %d.   b index %d  \n',ii,b_ind);
            % train
            if strcmp(alg,'CB')||strcmp(alg,'AROW') 
                res_run = shampo( data1, params );
                w = res_run.w;
                v = res_run.v;
                % test
                for jj=1:num_tasks,
                    % task,b value,perceptron method
                    test_res(jj, b_ind, 1) = mean( ((w{jj}' * data1{jj}.test.x) .* data1{jj}.test.y) <=0);
                    test_res(jj, b_ind, 2) = mean( ((v{jj}' * data1{jj}.test.x) .* data1{jj}.test.y) <=0);
                end
            elseif (strcmp(alg,'SO'))
                res_run = second_order_shampo_diag( data1, params );
                w = res_run.w;
                invA = res_run.invA;
                 % test
                for jj=1:num_tasks,
                    % task,b value,perceptron method
                    test_res(jj, b_ind) = mean( ((data1{jj}.test.x)'*((invA{jj}).*w{jj}).*data1{jj}.test.y') <=0);
                end
            elseif (strcmp(alg,'AROW'))
                res_run = second_order_shampo_diag( data1, params );
                w = res_run.w;
                %invA = res_run.invA;
                for jj=1:num_tasks
                    test_res(jj, b_ind) =  mean(((w{jj}' * data1{jj}.test.x) .* data1{jj}.test.y) <=0);
                end
            end
            queried(:,b_ind) = res_run.queried;
            train_err_total(:,b_ind) = res_run.train_err_total;
            train_err_queried(:,b_ind) = res_run.train_err_queried;

        end
        %----------------------------------------------------
    end
        all_res{ii}.test_res = test_res;
        all_res{ii}.train_err_total = train_err_total;
        all_res{ii}.train_err_queried = train_err_queried;
        all_res{ii}.queried = queried;
    if (ii==n_trials)%((mod(ii, save_every)) == 0)||(&&(n_trials<save_every)),
        fprintf('saving... %d  %s\n',ii,fn);
        save(sprintf('../results/%s',fn),'all_res','params');
    end
end

clear data;
clear data1;
