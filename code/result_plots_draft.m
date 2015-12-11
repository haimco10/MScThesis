% scatch for results
n_b = size(all_res,2);
n_t = size(all_res{1}.margin,2);
K = size(all_res{1}.margin,1);
Tt = 1:n_t;
TD_t = zeros(n_b,n_t);
Tb = zeros(n_b,n_t);

for idx = 1:n_b
    margin = abs(all_res{idx}.margin);
    b = all_res{idx}.b;
    minmargin = min(margin);
    margdiff = zeros(size(margin));
    
    for ii =1:size(margin,1)
        margdiff (ii,:) = margin(ii,:)-minmargin;
    end
    prob = 1./(b+margdiff);
    D_t = sum(prob);
    Tb(idx,:) = b*ones(1,n_t);
    TD_t(idx,:) = D_t;
end

% 3D plot is not informative

% figure;mesh(repmat(Tt,n_b,1),Tb,1./TD_t); 
% hold on;
% mesh(repmat(Tt,n_b,1),Tb,Tb/K);
figure;plot(1./TD_t(:,20),Tb(:,1));
hold on
plot(Tb(:,20)/K,Tb(:,1),'r');
set(gca,'XScale','log')
set(gca,'YScale','log')



figure;plot(1./TD_t(10,10:end));
hold on
plot(b*ones(1,length(D_t))./K,'r')



figure;
plot(cumsum(all_res{13}.queried(1:5000) ==1)','r');
title('queried 1,2')
hold on;
plot(cumsum(all_res{13}.queried(1:5000) ==2)','b');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
plot(cumsum(all_res{1}.train_error'))
legend('0','1','2','3','4','5','6','7','8','9');
title('train errors -1');

figure;
plot(cumsum(all_res{6}.train_error'))
legend('0','1','2','3','4','5','6','7','8','9');
title('train errors -6');

figure;
plot(cumsum(all_res{13}.train_error'))
legend('0','1','2','3','4','5','6','7','8','9');
title('train errors -13');

figure;
plot(cumsum(all_res{13}.test_res'))
title('test errors');
legend('0','1','2','3','4','5','6','7','8','9');

%%%%%%%%%%%%
figure;
trainErr = all_res{6}.train_error();
plot(all_res{6}.train_error()')








