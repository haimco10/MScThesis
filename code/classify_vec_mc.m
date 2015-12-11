%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% runs online linear (first order) classifiers on training data
% input:
%   - data : training data
%           - x        : each instance is a column vector
%           - y        : each column is an indicator of possible labels
%   - params :
%           - n_runs   : number of iterations over the data
%           - type     : algorithm 'perceptron' 'pa'
%           - C        : parameter for mira
%
% output:
%   - res :
%           - mu       : weight paramters
%           - avgmu    : average weight paramters
%           - pred     : score using mu models
%           - avgpred  : score using avgmu models
%           - erros    : error indicator using mu model
%           - avgerros : error indicator using avgmu model
%
%
% CW_1.0 - Confidence weighted learning, Version 1.0
%
% See:
% (1) Confidence-Weighted Linear Classification
%     Mark Dredze, Koby Crammer and Fernando Pereira
%     Proceedings of the 25th International Conference on Machine Learning (ICML), 2008 
%     http://webee.technion.ac.il/people/koby/publications/icml08_variance.pdf
% (2) Exact Convex Confidence-Weighted Learning
%     Koby Crammer, Mark Dredze and Fernando Pereira
%     Proceedings of the Twenty Second Annual Conference on Neural
%     Information Processing Systems (NIPS), 2008 
%     http://webee.technion.ac.il/people/koby/publications/paper_nips08_std.pdf
% (3) Multi-Class Confidence Weighted Algorithms
%     Koby Crammer, Mark Dredze and Alex Kulesza
%     Empirical Methods in Natural Language Processing (EMNLP), 2009 
%     http://webee.technion.ac.il/people/koby/publications/mccw_emnlp09.pdf
% (4) Adaptive Regularization Of Weight Vectors
%     Koby Crammer, Alex Kulesza and Mark Dredze
%     Advances in Neural Information Processing Systems, 2009
%     http://webee.technion.ac.il/people/koby/publications/arow_nips09.pdf
%
%
%    This program is free software; you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation; either version 2 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program; if not, write to the Free Software
%    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
%
% Written by Koby Crammer, 2010
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res = classify_vec_mc(data, params);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extract data information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = data.x; y = data.y;
n_feat = size(x,1);
n_examples = size(x,2);
n_label = size(y,1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu    = zeros(n_feat, n_label);
avgmu = mu;

% predictions
pred      = zeros(1, n_examples * params.n_runs);
avgpred   = zeros(1, n_examples * params.n_runs);
errors    = zeros(1, n_examples * params.n_runs);
avgerrors = zeros(1, n_examples * params.n_runs);

% index of current example ( over n_runs and examples )
ind = 1;

% iterate over the training set
n_iters = params.n_runs * n_examples;
for run_i = 1:params.n_runs,
    
    % iterate over exampels
    for round = 1:n_examples,
        
        % extract current input and label
        x1 = x(:, round);
        y1 = y(:, round);
        y1_in  = (y1);
        y1_out = (~y1);
        
        % last model
        %scores = mu' * x1;
        scores = (x1'*mu)';
        
        [hp, yh] = max(scores);
        pred(ind) = yh;
        scores_in  = scores; scores_in(y1_out) = -inf;
        scores_out = scores; scores_out(y1_in) = -inf;
        
        [label_score,           label] = max(scores_in);
        [competitor_score, competitor] = max(scores_out);
        errors(ind) = yh ~= label;
        
        % avg model
        ascores = (x1'*avgmu)';
        [ahp, ayh] = max(ascores);
        avgpred(ind) = ayh;
        ascores_in  = ascores; ascores_in(y1_out) = -inf;
        ascores_out = ascores; ascores_out(y1_in) = -inf;
        
        [label_ascore, alabel]           = max(ascores_in);
        [competitor_ascore, acompetitor] = max(ascores_out);
        avgerrors(ind) = ayh ~= alabel;
        
        % update
        % extract top label in/out
        
        margin = label_score - competitor_score;
        
        switch (params.type)
            case ('perceptron')
                
                if (margin <= 0)
                    mu(:, label)      = mu(:, label)      + x1;
                    mu(:, competitor) = mu(:, competitor) - x1;
                    
                    avgmu(:, label)      = avgmu(:, label)      + (n_iters - ind + 1) * x1;
                    avgmu(:, competitor) = avgmu(:, competitor) - (n_iters - ind + 1) * x1;
                end
                
                
            case ('pa')
                if (margin < 1)
                    alpha = (1-margin)/(x1' * x1);
                    if (alpha > params.C)
                        alpha = params.C;
                    end
                    
                    mu(:, label)      = mu(:, label)      + alpha * x1;
                    mu(:, competitor) = mu(:, competitor) - alpha * x1;
                    
                    avgmu(:, label)      = avgmu(:, label)      + alpha * (n_iters - ind + 1) * x1;
                    avgmu(:, competitor) = avgmu(:, competitor) - alpha * (n_iters - ind + 1) * x1;
                end
        end
        %avgmu = avgmu + mu;
        
        ind = ind + 1;
        
        
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
res.mu       = mu;
res.avgmu    = avgmu;
res.pred      = pred;
res.avgpred   = avgpred;
res.errors    = errors;
res.avgerrors = avgerrors;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
