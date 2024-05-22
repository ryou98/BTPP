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


x = 0:300:300*(length(pushpull.train_loss)-1);


% Create the plot
figure;
plot(x, pushpull.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', 'r','Marker', 'o',  'MarkerSize',2)



hold on
grid on

plot(x, dsgd_ring.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0, 0.7, 0.7],'Marker', 'diamond', 'MarkerSize',2)

plot(x, dsgt_ring.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.3, 0.5, 0],'Marker', '^', 'MarkerSize',2)

plot(x, dsgd_onepeerexp.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.6, 0.4, 0],'Marker', '+', 'MarkerSize',2)

plot(x, dsgd_ceca2p.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.9, 0.3, 0.9],'Marker', '^', 'MarkerSize',2)

plot(x, dsgd_base_k.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.3, 0.2, 0.2],'Marker', 'diamond', 'MarkerSize',2)

% plot(x, dsgd_odeq.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
%     'Color', [0.3, 0.7, 0.2],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgt_odeq.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.3, 0.5, 0.8],'Marker', '+', 'MarkerSize',2)

plot(x, relaysgd_btree.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.4, 0.7, 0.9],'Marker', '^', 'MarkerSize',2)

plot(x, dsgd_d2_ring.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.4, 0.3, 0.9],'Marker', 'o', 'MarkerSize',2)

plot(x, dsgd_exp.train_loss, 'LineStyle', '--', 'LineWidth', 1, ...
    'Color', [0.1, 0.1, 0.9],'Marker', '+', 'MarkerSize',2)

plot(x, dsgd_fully.train_loss, 'LineStyle', '--', 'LineWidth', 1, ...
    'Color', [0.1, 0.1, 0.5],'Marker', 'diamond', 'MarkerSize',2)

plot(x, dsgd_grid.train_loss, 'LineStyle', '-', 'LineWidth', 1, ...
    'Color', [0.5, 0.5, 0.5],'Marker', '^', 'MarkerSize',2)

% title('Train Loss - Data Heterogeneity');
title('Train Loss')
xlabel('Iteration');
ylabel('Loss');


% Add a legend
legend( 'BTPP',...
    'DSGD-Ring', ...
    'DSGT-Ring', ...
    'DSGD-OnePeerExp', ...
    'DSGD-CECA-2p', ...
    'DSGD-Base-(k+1)', ... %'DSGD-ODEqui', ...
    'DSGT-ODEquiDyn', ...
    'RelaySGD-B-Tree', ...
    'D2-Ring', ...
    'DSGD-Exponential',...
    'DSGD-FullyConnected',...
    'DSGD-Grid',...
    'Location', 'northeast');


% Adjust the size of the figure (optional)
set(gcf, 'Position', [100, 100, 600, 400]); % [x y width height]

% Save the plot
saveas(gcf, 'loss_cifar10.png'); % Saves the figure as a PNG file