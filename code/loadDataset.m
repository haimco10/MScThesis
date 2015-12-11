function [data,n_tasks,n_ex] = loadDataset(name)

data = [];

my_dir = '/Users/haimcohen/Documents/privateDocs/ThesisData/data';
if strcmp(name,'usps1rest')
    load (sprintf('%s/%s',my_dir,'usps_onerest.mat'));
end

n_tasks = length(data);
n_ex = size(data{1}.train.x,2);
for jj=1:n_tasks
    n_ex = min([n_ex,size(data{jj}.train.x, 2)]);
end

end