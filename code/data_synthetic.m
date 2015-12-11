% data = data_synthetic(n_examples, params)
%
% input:
% - n_examles    : number of examples
% params:
% - margin       : minimal margin between two classes
% - margin_noise : amount of margin noise
% - n_noisy_feat : number of noisy features
% - noise        : std of noisy features
% - label_noise  : amout of value noise
% 
% generates two features that correspnd to a 45 degrees roated 
% gaussian with axes sizes 1 and 10
% and additional randaom noisy featues
%
% Written by Koby Crammer, (c) 2010
%
function data = data_synthetic(n_examples, params)
margin = params.margin;
margin_noise = params.margin_noise;
n_noisy_feat = params.n_noisy_feat;
noise = params.noise;

if (~isfield(params,'label_noise'))
  params.label_noise = 0;
end
label_noise = params.label_noise;

%%%%%%%%%%%%%%%%%%%%%%%%%

n_feat = n_noisy_feat + 2;
x = randn(n_feat, n_examples);
i=find(abs(x(2,:))<margin);
x(2,i) = margin .* sign(x(2,i));

y = sign(x(2,:));
y(find(y==0))=1;

x(1,:) = x(1,:) * 10;
x(3:end,:) = x(3:end,:) * noise;
for j=3:n_feat,
    i=find(abs(x(j,:))<margin_noise);
    x(j,i) = margin_noise .* sign(x(j,i));
end

angle = 45 / (2*pi);
x(1:2,:) = [cos(angle),sin(angle); -sin(angle),cos(angle)] * x(1:2,:);

i = rand(n_examples, 1) < label_noise;
y(i) = -y(i);


data.x = x;
data.y = y;
data.n = i';
