% data = generate_data_synthetic(params)
%
% generates train and test set of given size
% Written by Koby Crammer, (c) 2009
%

function data = generate_data_synthetic(params)

if isfield(params, 'true_random')
  disp('1');
  rand('state',0);  randn('state',0);
end

data.train = data_synthetic(params.n_train, params);
data.train.y1= data.train.y;
data.train.y = mc_to_smc(data.train.y);

data.test  = data_synthetic(params.n_test, params);
data.test.y1= data.test.y;
data.test.y = mc_to_smc(data.test.y);


function y1 = mc_to_smc(y)
n=length(y);
y1 = spalloc( 2, n, n) ;

y1(1, (y == -1))=1;
y1(2, (y == +1))=1;
y1 = logical(y1);