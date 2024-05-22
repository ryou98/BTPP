% load data

pushpull = load('1pushpullcifar10resnet188.mat');
dsgd_ring = load('1dsgd_staticringcifar10resnet188.mat');
dsgt_ring = load('1dsgt_staticringcifar10resnet188.mat');
dsgd_onepeerexp = load('1dsgd_dynamicOnePeerExpcifar10resnet188.mat');
dsgd_ceca2p = load('1dsgd_cecaceca-2pcifar10resnet188.mat');
dsgd_base_k = load('1dsgd_dynamicbase_kcifar10resnet188.mat');
dsgt_odeq = load('1dsgt_dynamicODEquiDyncifar10resnet188.mat');
relaysgd_btree = load('1relaysgdrelay_binarytreecifar10resnet188.mat');
dsgd_d2_ring = load('1dsgd_d2ringcifar10resnet188.mat');
dsgd_exp = load('1dsgd_staticexponentialcifar10resnet188.mat');
dsgd_fully = load('1dsgd_staticfully_connectedcifar10resnet188.mat');
dsgd_grid = load('1dsgd_staticgridcifar10resnet188.mat');

x = 0:300:300*(length(pushpull.test_acc)-1);


% Create the plot
figure;
plot(x, pushpull.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', 'r','Marker', '^',  'MarkerSize',2)



hold on
grid on

plot(x, dsgd_ring.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0, 0.7, 0.7],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgt_ring.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.3, 0.5, 0],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgd_onepeerexp.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.6, 0.4, 0],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgd_ceca2p.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.9, 0.3, 0.9],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgd_base_k.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.3, 0.2, 0.2],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgt_odeq.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.3, 0.5, 0.8],'Marker', 'o', 'MarkerSize',2)

plot(x, relaysgd_btree.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.4, 0.7, 0.9],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgd_d2_ring.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.4, 0.3, 0.9],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgd_exp.test_acc, 'LineStyle', '--', 'LineWidth', 1, ...
    'Color', [0.1, 0.1, 0.9],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgd_fully.test_acc, 'LineStyle', '--', 'LineWidth', 1, ...
    'Color', [0.1, 0.1, 0.5],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgd_grid.test_acc, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.5, 0.5, 0.5],'Marker', 'o', 'MarkerSize',2)

% title('Test Accuracy - Data Heterogeneity');
title('Test Accuracy');
xlabel('Iteration');
ylabel('Accuracy (%)');


% Add a legend
legend('BTPP',...
    'DSGD-Ring', ...
    'DSGT-Ring', ...
    'DSGD-OnePeerExp', ...
    'DSGD-CECA-2p', ...
    'DSGD-Base-(k+1)', ... 
    'DSGT-ODEquiDyn', ...
    'RelaySGD-B-Tree', ...
    'D2-Ring', ...
    'DSGD-Exponential',...
    'DSGD-FullyConnected',...
    'DSGD-Grid',...
    'Location', 'southeast');


% Adjust the size of the figure (optional)
set(gcf, 'Position', [100, 100, 600, 400]); % [x y width height]

% Save the plot
saveas(gcf, 'acc_cifar10.png'); % Saves the figure as a PNG file