function [data] = loadData( data_name )
%GETDATA load and get the data for the algorithm
%   The finction return the data as a... to do...
    
 if strcmp(data_name,'vj_onerest')
     my_dir = '../data';
     load (sprintf('%s/%s',my_dir,'VJ_onerest.mat'));
 elseif strcmp(data_name,'vj_pairs')
     my_dir = '../data';
     load (sprintf('%s/%s',my_dir,'VJ_pairs.mat'));  
 elseif strcmp(data_name,'mnist_onerest')
     my_dir = '../data';
     load (sprintf('%s/%s',my_dir,'mnist_onerest.mat'));     
 elseif strcmp(data_name,'mnist_pairs')
     my_dir = '../data';
     load (sprintf('%s/%s',my_dir,'mnist_pairs.mat')); 
 elseif strcmp(data_name,'usps_onerest')
     my_dir = '../data';
     load (sprintf('%s/%s',my_dir,'usps_onerest.mat'));     
 elseif strcmp(data_name,'usps_pairs')
     my_dir = '../data';
     load (sprintf('%s/%s',my_dir,'usps_pairs.mat')); 
 elseif strcmp(data_name,'NLP')
     my_dir = '../data';
     load (sprintf('%s/%s',my_dir,'NLP.mat')); 
      elseif strcmp(data_name,'mixed')
     my_dir = '../data';
     load (sprintf('%s/%s',my_dir,'MIXED.mat')); 
% TODO:
%  elseif strcmp(data_name,'...')
%      my_dir = '/scratch/koby/corpora/all_binary/data/data/'
%      load_data_...;
 end
end