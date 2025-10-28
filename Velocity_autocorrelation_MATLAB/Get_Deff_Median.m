function [Dmed, Dlin, DSEM, names, MSD_ensemble] = Get_Deff_Median(Complex)  
%Example of using is "[Dmed, Dlin, DSEM, names, MSD_ensemble] = Get_Deff_Median(1)" 
%MSD_ensemble is the ensemble & time averaged MSD for every condition (also averaged among images within each condition)
    % note - should pass MSD forward next
    if nargin <1 
        Complex = 0;
        Min_Deff = 0;
        Max_Deff = 10;
        V3 = 0;
        Min_Length = 0;
        Max_Length = 1000;
        fprintf('Assuming no complex capabilities. If you want to change plotting or use V3 \ncapabilities, run script with any value, for example, run: "Get_Deff_Median(1)"  ');
    else
        prompt = {'Min effective diffusion constant for plot','Max effective diffusion constant for plot', 'Enable V3 capabilities?', 'Minimum track length for plot (>= 11 is use for Deff analysis and is recommended','Max track length for plot'};
        title = 'Settings for results plots';
        dims = [1 35];
        definput = {'0','10','1','11','100'};
        settings  = inputdlg(prompt,title,dims,definput);
        Min_Deff = str2double(settings{1});
        Max_Deff = str2double(settings{2});
        V3 = str2double(settings{3});
        Min_Length = str2double(settings{4});
        Max_Length = str2double(settings{5});
    end
        


%Navigate to a directory with only your .mat results files and nothing
%else then grab the names with the following code
% should add # trajectories

[filename,path] = uigetfile('multiselect','on','.mat','Select the file to convert');
 cd(path)

if Complex ~= 0
    alltracklengths = cell(length(filename),1);
% Next extract the tracks from these result files with one of the next two loops.
    for i = 1:length(filename)
        disp(filename{i})
        result = importdata(filename{i});
        Dlin{i} = result.lin.D_lin{1,1};
        Dmed(i) = median(result.lin.D_lin{:});
        DSEM(i) = std(result.lin.D_lin{:})/sqrt(length(result.lin.D_lin{:}));
        
        MSD_ensemble{i} = result.MSD.MSD_time{1,1};
        alltracklengths{i} = result.lin.track_length{1,1};
        
        %these are pretty wasteful, but I am keeping because I am lazy
        tracklength_tot_med(i) = median(result.lin.track_length{1,1});
        tracklength_tot_mean(i) = mean(result.lin.track_length{1,1});
        tracklength_tot_SEM(i) = std(result.lin.track_length{1,1})/sqrt(length(result.lin.track_length{1,1}));

        tracklength_range_med(i) = median(result.lin.track_length{1,1}(result.lin.track_length{1,1} >= 11));
        tracklength_range_mean(i) = mean(result.lin.track_length{1,1}(result.lin.track_length{1,1} >= 11));
        tracklength_range_SEM(i) = std(result.lin.track_length{1,1}(result.lin.track_length{1,1} >= 11)) / sqrt(length(result.lin.track_length{1,1}(result.lin.track_length{1,1} >= 11)));
    end
else%To include all tracks
    for i = 1:length(filename)
        disp(filename{i})
        result = importdata(filename{i});
        Dlin{i} = result.lin.D_lin{1,1};
        Dmed(i) = median(result.lin.D_lin{:});
        DSEM(i) = std(result.lin.D_lin{:})/sqrt(length(result.lin.D_lin{:}));

    end
end
% When you want to cut off tracks that don't move
% for i = 1:length(a)
%     disp(a{i})
%     result = importdata(a{i});
%     cut =  result.lin.D_lin{:} > 0.0095 & result.lin.D_lin{:} < 0.08; %result.lin.D_lin{:} > 0.001 &
%     cut_data = result.lin.D_lin{:}(cut)
%     med(i) = median(cut_data);
%     SEM(i) = std(result.lin.D_lin{:})/sqrt(length(result.lin.D_lin{:}));
% end

% run one of the two loops above then create the plot with the
% following code

%% 

% will want to plot speed, distribution of speeds, various track lengths +
% SEM, fraction of track lenghts above / below thresholds, histogram of
% track lengths, next step correlation, anomalous coefficient
h = figure;
hold on
scatter(1:length(filename),Dmed,'LineWidth',3)
errorbar(1:length(filename),Dmed,DSEM,'.','LineWidth',3)
names = filename';

set(gca, 'XTick', 1:length(names),'XTickLabel',names, 'FontSize', 20);
ylabel( 'Deff (\mum^2.s^{-1})','FontSize', 20);
set(gca,'xticklabel',names,'FontSize', 20, 'TickLabelInterpreter', 'None')
set(gca,'XTickLabelRotation',45)
pbaspect([1 1 1])
%% 
p = figure;
hold on
set(gca,'DefaultLineLineWidth',3)
for  i = 1:length(filename)
    cdfplot(Dlin{i})
end

names = filename';

legend(names,'Location','best','FontSize', 20,'Interpreter','None')
xlim([0,6])
ylabel( 'CDF Deff','FontSize', 20);
xlabel( 'Deff (\mum^2.s^{-1})','FontSize', 20);
pbaspect([1 1 1])
hold off
%% 
r = figure;
hold on

for  i = 1:length(filename)
    [N_single,edges_single] = histcounts(Dlin{i},'BinWidth', 0.05);
% find limits from each histogram for a diffusion range
    N{i} = N_single;
    edges{i} = edges_single;
end

maxtot = max(cellfun(@(x) max((x)),N)); % Am I sure I am calculating the trapezoid correctly?
for r = 1:length(N)
    E = edges{r};
    E = E(1:length(edges{r})-1);
    H = N{r} / trapz(0.1,N{r});
    H = N{r} / maxtot;
    histyvalue = H(E >= 0 & E <= 4);
    histxvalue = E(E >= 0 & E <= 4);
    plot(histxvalue,histyvalue,'LineWidth',3)
end
    
legend(names,'Location','best','FontSize', 20,'Interpreter','None')
%xlim([0,6])
ylabel( 'Frequency','FontSize', 20);
xlabel( 'Deff (\mum^2.s^{-1})','FontSize', 20);
pbaspect([1 1 1])
hold off

%% 

% j = figure
% hold on
% set(gca,'DefaultLineLineWidth',3)
% scatter(1:length(filename),tracklength_tot_med);
% errorbar(1:length(filename),tracklength_tot_med,tracklength_tot_SEM,'.','LineWidth',3)
% names = filename;
% names = filename'
% 
% set(gca, 'XTick', 1:length(names),'XTickLabel',names,'FontSize', 20);
% ylabel( 'Median tracklength all tracks','FontSize', 20);
% set(gca,'xticklabel',names,'FontSize', 20)
% set(gca,'XTickLabelRotation',45)
% 
% %% 
% % 
% k = figure
% hold on
% scatter(1:length(filename),tracklength_range_med,'LineWidth',3)
% errorbar(1:length(filename),tracklength_range_med,tracklength_range_SEM,'.','LineWidth',3)
% names = filename;
% names = filename'
% 
% set(gca, 'XTick', 1:length(names),'XTickLabel',names,'FontSize', 20);
% ylabel( 'Median tracklength tracks > 11 frames','FontSize', 20);
% set(gca,'xticklabel',names,'FontSize', 20)
% set(gca,'XTickLabelRotation',45)
% %clearvars -except Dmed Dlin

%% 
q = figure;
hold on
clear i

for  i = 1:length(filename)
    [length_counts_single,length_edges_single] = histcounts(alltracklengths{i},'BinWidth', 5);
% find limits from each histogram for a diffusion range
   length_edges_single = length_edges_single(1:length(length_edges_single)-1);
    N_length{i} = length_counts_single;
    edges_length{i} = length_edges_single;
end


for r = 1:length(N_length)
    plot(edges_length{r},N_length{r},'LineWidth',3)
end
    
legend(names,'Location','best','FontSize', 20,'Interpreter','None')
xlim([0,50])
set(gca, 'YScale', 'log')
ylabel( 'Frequency','FontSize', 20);
xlabel( 'Track Length','FontSize', 20);
pbaspect([1 1 1])
hold off
%% 
w = figure;
hold on
% should incorporate error bars into MSD plotting?
for r = 1:length(filename)
    plot(MSD_ensemble{r},'LineWidth',3)
end
% next up try adding corral analysis?
% https://www.nature.com/articles/srep34987#ref22 https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.4915
end

