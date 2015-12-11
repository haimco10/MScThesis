% Shampo tests


clear all; close all; clc;
%dbstop if all error

%% in order to allow parfor
pools = matlabpool('size');
cpus = feature('numCores');
if (pools~=0)
    matlabpool('close');
end
matlabpool('open',cpus-1);


% pools = matlabpool('size');
% cpus = feature('numCores');
% if pools ~= (cpus - 1)
%     if pools > 0
%         matlabpool('close');
%     end
%     matlabpool('open', cpus - 1);
% end

%%

%alg = 'SO';
alg = 'AROW';
c = 0.001;
update = 'aggressive';

%dataset = 'usps_pairs';
dataset = 'mnist_pairs';
%dataset = 'vj_pairs';
%dataset = 'usps_onerest';
%dataset = 'mnist_onerest';
%dataset = 'vj_onerest';
%dataset = 'NLP';

n_hard_from = 0.6;
n_easy_to = 0.4;
min_examples = 1e6;
b_all = 10.^(-7:5);
n_trials = 20;

max_n_tasks = 6;


for num_tasks = 2:max_n_tasks
    parfor n_easy = 1:(num_tasks-1)
        n_hard = num_tasks-n_easy;
        aggressive_th = 0;
        run_shampo_diag(n_hard,n_easy,n_hard_from,n_easy_to,min_examples,...
            b_all,aggressive_th,n_trials,dataset,alg,update,c);
    end
end





