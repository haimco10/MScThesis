%% generate data
params_data.margin = 0.01;
params_data.margin_noise = -1;
params_data.n_noisy_feat = 18;
params_data.noise =8.5;
params_data.label_noise = 0.1;

params_data.n_train = 5000;
params_data.n_test = 10000;

data_noise = generate_data_synthetic(params_data);

params_data.margin = 0.01;
params_data.margin_noise = -1;
params_data.n_noisy_feat = 18;
params_data.noise =8.5;
params_data.label_noise = -0.1;

params_data.n_train = 5000;
params_data.n_test = 10000;

data = generate_data_synthetic(params_data);
data.train = data_noise.train;
fprintf('---------------------------------------------------\n');
fprintf('algorithm\t\t\t\ttest error\n');
fprintf('---------------------------------------------------\n');


%% plot data
figure
hold on; grid on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Perceptron
%%%%%%%%%%%%%%%%
params.n_runs = 1;
params.type = 'perceptron';
res_perceptron = classify_vec_mc(data.train, params);
fprintf('perceptron          \t\t\t%5.2f\n',100*mean( (diff(res_perceptron.mu')*data.test.x).*data.test.y1 <=0));
fprintf('avg perceptron      \t\t\t%5.2f\n',100*mean( ...
    (diff(res_perceptron.avgmu')*data.test.x).*data.test.y1 <=0));
errorplot( res_perceptron.errors, data.train.n, 'r');
