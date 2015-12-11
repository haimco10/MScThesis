%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% runs online CW classifiers on training data
% input:
%   - data : training data
%           - x        : each instance is a column vector
%           - y        : each column is an indicator of possible labels
%   - params :
%           - n_runs   : number of iterations over the data
%           - type     : algorithm 'perceptron' 'mira'
%           - C        : parameter for CW/AROW
%
% output:
%   - res :
%           - mu       : weight paramters
%           - avgmu    : average weight paramters
%           - pred     : score using mu models
%           - avgpred  : score using avgmu models
%           - erros    : error indicator using mu model
%           - avgerros : error indicator using avgmu model
%           - variance : variance in margin
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
% Written by Koby Crammer, (c) 2010
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function res = classify_mat_diag_mc(data, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isfield(params,'sparse')
    params.sparse = 1;
end


if isfield(params,'C')
    params.q           = params.C;
    params.inv_erf_eta = params.C;
end

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
sigma    = ones(n_feat, n_label);
avgmu = mu;


% predictions
pred      = zeros(1, n_examples * params.n_runs);
avgpred   = zeros(1, n_examples * params.n_runs);
errors    = zeros(1, n_examples * params.n_runs);
avgerrors = zeros(1, n_examples * params.n_runs);
variance  = zeros(1, n_examples * params.n_runs);

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
        scores = (x1' * mu)';
        [hp, yh] = max(scores);
        pred(ind) = yh;
        scores_in  = scores; scores_in(y1_out) = -inf;
        scores_out = scores; scores_out(y1_in) = -inf;

        [label_score,           label] = max(scores_in);
        [competitor_score, competitor] = max(scores_out);
        errors(ind) = yh ~= label;

        % avg model
        ascores = (x1' * avgmu)';
        [ahp, ayh] = max(ascores);
        avgpred(ind) = ayh;
        ascores_in  = ascores; ascores_in(y1_out) = -inf;
        ascores_out = ascores; ascores_out(y1_in) = -inf;

        [label_ascore, alabel]           = max(ascores_in);
        [competitor_ascore, acompetitor] = max(ascores_out);
        avgerrors(ind) = ayh ~= alabel;

        % update
        % extract top label in/out
        nnz_inds = logical((x1));

        sigma_x  = sigma .* repmat(x1    ,1 ,n_label);
        sigma_x2 = sigma .* repmat(x1.^2 ,1 ,n_label);

        x_sigma_x_vec = sum(sigma_x2);

        margin = scores(label) - scores(competitor);
        x_sigma_x = x_sigma_x_vec(label) + x_sigma_x_vec(competitor);
        variance(ind) = x_sigma_x;

        switch (params.type)
           
            case ('arow_project_dense')
                % nips 2009
                if (margin <= 1)
                    c = x_sigma_x + params.q;
                    alpha = 1 - margin;
                    mu(:,label)      = mu(:,label)      +  (sigma_x(:,label))     *(alpha/c);
                    mu(:,competitor) = mu(:,competitor) -  (sigma_x(:,competitor))*(alpha/c);

                    sigma(:,label)      = 1./(1./sigma(:,label)      + x1.^2 / params.q);
                    sigma(:,competitor) = 1./(1./sigma(:,competitor) + x1.^2 / params.q);
                end


            case ('arow_project')
                % nips 2009

                if (margin <= 1)
                    c = x_sigma_x + params.q;
                    alpha = 1 - margin;
                    mu(:,label)      = mu(:,label)      +  (sigma_x(:,label))     *(alpha/c);
                    mu(:,competitor) = mu(:,competitor) -  (sigma_x(:,competitor))*(alpha/c);
                    avgmu(:,label)      = avgmu(:,label)      + (n_iters - ind + 1) * (sigma_x(:,label))     *(alpha/c);
                    avgmu(:,competitor) = avgmu(:,competitor) - (n_iters - ind + 1) * (sigma_x(:,competitor))*(alpha/c);

                    sigma(nnz_inds,label)      = 1./(1./sigma(nnz_inds,label)      + x1(nnz_inds).^2 / params.q);
                    sigma(nnz_inds,competitor) = 1./(1./sigma(nnz_inds,competitor) + x1(nnz_inds).^2 / params.q);

                end

            case ('arow_drop')
                % nips 2009
                if (margin <= 1)
                    c = x_sigma_x + params.q;
                    alpha = 1 - margin;
                    mu(:,label)      = mu(:,label)      +  (sigma_x(:,label))     *(alpha/c);
                    mu(:,competitor) = mu(:,competitor) -  (sigma_x(:,competitor))*(alpha/c);

                    sigma(:,label)      = (sigma(:,label)      - (sigma_x(:,label).^2     )/c) ;
                    sigma(:,competitor) = (sigma(:,competitor) - (sigma_x(:,competitor).^2)/c) ;
                end


            case ('std_project')
                % nips 2008
                if (margin <= params.inv_erf_eta * sqrt(x_sigma_x))
                    phi = params.inv_erf_eta;

                    opp2 = 1 + phi^2;
                    opp2h = 1 + 0.5*phi^2;
                    term1 = -margin * opp2h;
                    term2 = opp2 * (margin^2 - x_sigma_x * (phi^2));
                    alpha = (term1 + sqrt((term1)^2 - term2)) / ( x_sigma_x * opp2 );
                    term3 = -alpha * x_sigma_x * phi;
                    term4 = (( term3 + sqrt( (term3)^2 + 4 * x_sigma_x))/2)^2; % u
                    %beta = (alpha * phi)/(sqrt(term4) + (alpha*phi*x_sigma_x));

                    mu(:,label)      = mu(:,label)      +  (sigma_x(:,label))     *(alpha);
                    mu(:,competitor) = mu(:,competitor) -  (sigma_x(:,competitor))*(alpha);

                    sigma(:,label)      = 1./(1./sigma(:,label)      + alpha*phi/sqrt(term4)*x1.^2);
                    sigma(:,competitor) = 1./(1./sigma(:,competitor) + alpha*phi/sqrt(term4)*x1.^2);

                end

            case ('std_drop')
                % nips 2008
                if (margin <= params.inv_erf_eta * sqrt(x_sigma_x))
                    phi = params.inv_erf_eta;

                    opp2 = 1 + phi^2;
                    opp2h = 1 + 0.5*phi^2;
                    term1 = -margin * opp2h;
                    term2 = opp2 * (margin^2 - x_sigma_x * (phi^2));
                    alpha = (term1 + sqrt((term1)^2 - term2)) / ( x_sigma_x * opp2 );
                    term3 = -alpha * x_sigma_x * phi;
                    term4 = (( term3 + sqrt( (term3)^2 + 4 * x_sigma_x))/2)^2; % u
                    beta = (alpha * phi)/(sqrt(term4) + (alpha*phi*x_sigma_x));

                    mu(:,label)      = mu(:,label)      +  (sigma_x(:,label))     *(alpha);
                    mu(:,competitor) = mu(:,competitor) -  (sigma_x(:,competitor))*(alpha);

                    sigma(:,label)      = (sigma(:,label)      - (sigma_x(:,label).^2     ) * beta) ;
                    sigma(:,competitor) = (sigma(:,competitor) - (sigma_x(:,competitor).^2) * beta) ;

                end


            case ('variance_drop')
                % icml 2008
                if (margin <= params.inv_erf_eta * (x_sigma_x))
                    phi = params.inv_erf_eta;

                    term1 = 1+2*phi*margin;
                    term2 = margin-phi*x_sigma_x;
                    alpha = (-term1 + sqrt(term1^2 - 8*phi*term2))/(4*phi*x_sigma_x);
                    beta = (2*alpha*phi)/(1+2*alpha*phi*x_sigma_x);

                    mu(:,label)      = mu(:,label)      +  (sigma_x(:,label))     *(alpha);
                    mu(:,competitor) = mu(:,competitor) -  (sigma_x(:,competitor))*(alpha);

                    sigma(:,label)      = (sigma(:,label)      - (sigma_x(:,label).^2     ) * beta) ;
                    sigma(:,competitor) = (sigma(:,competitor) - (sigma_x(:,competitor).^2) * beta) ;

                end



            case ('variance_project')
                % icml 2008
                if (margin <= params.inv_erf_eta * (x_sigma_x))
                    phi = params.inv_erf_eta;

                    term1 = 1+2*phi*margin;
                    term2 = margin-phi*x_sigma_x;
                    alpha = (-term1 + sqrt(term1^2 - 8*phi*term2))/(4*phi*x_sigma_x);

                    mu(:,label)      = mu(:,label)      +  (sigma_x(:,label))     *(alpha);
                    mu(:,competitor) = mu(:,competitor) -  (sigma_x(:,competitor))*(alpha);

                    sigma(:,label)      = 1./(1./sigma(:,label)      + 2 * alpha * params.inv_erf_eta * x1.^2);
                    sigma(:,competitor) = 1./(1./sigma(:,competitor) + 2 * alpha * params.inv_erf_eta * x1.^2);

                end
        end

        avgmu = avgmu + mu;
        ind = ind + 1;
    end
end

res.mu = mu;
res.avgmu = avgmu;

res.pred      = pred;
res.avgpred   = avgpred;
res.errors    = errors;
res.avgerrors = avgerrors;
res.variance = variance;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
