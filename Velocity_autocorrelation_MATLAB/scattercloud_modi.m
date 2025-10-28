function h = scattercloud_modi(x,y,n,l,clm)
%This script is modified from original "scattercloud.m" script located at 
%" /Users/tongshu/Documents/Lab project 2020/Nuclear GEM method paper/Partic
% le_tracking_for_alpha_D/scattercloud.m" 

%This script is used to plot droplet diffusivity versus droplet intensity
%in the log scale: x (intensity) vs y (diffusivity)
%or two fluorescent intensities in the log scale: x (intensity1) vs y (intensity2)
%Call this function using, where h can be used for legend:
% "h = scattercloud_modi(x_combined_larger_0,y_combined_larger_0,25,1,'r.')"
% "h = scattercloud_modi(scatter(:,1)-GFP_BG,scatter(:,2)-BFP_BG,25,1,'b.')"

%https://www.mathworks.com/matlabcentral/fileexchange/6037-scattercloud
%Example of using as:
%"scattercloud(log(Total_D_inst),Total_alpha,25,1,'k.','jet')"
%SCATTERCLOUD display density of scatter data
%   SCATTERCLOUD(X,Y) creates a scatterplot of X and Y, displayed over a
%   surface representing the smoothed density of the points.  The density is
%   determined with a 2D histogram, using 25 equally spaced bins in both
%   directions.
%   SCATTERCLOUD(X,Y,N) uses N equally spaced bins.
%   SCATTERCLOUD(X,Y,N,L) uses L as a parameter to the smoothing algorithm.
%    Defaults to 1.  Larger values of L lead to a smoother density, but a
%    worse fit to the original data.
%   SCATTERCLOUD(X,Y,N,L,CLM) uses CLM as the color/linestyle/marker for
%    the scatter plot.  Defaults to 'k+'.
%   SCATTERCLOUD(X,Y,N,L,CLM,CMAP) uses CMAP as the figure's colormap.  The
%    default is 'flipud(gray(256))'.
%   H = SCATTERCLOUD(...) returns the handles for the surface and line
%    objects created.
%
%   Example:
%
%     scattercloud(1:100 + randn(1,100), sin(1:100) + randn(1,100),...
%                  50,.5,'rx',jet(256))
% 
%   References: 
%     Eilers, Paul H. C. & Goeman, Jelle J. (2004). Enhancing scatterplots 
%   with smoothed densities. Bioinformatics 20(5), 623-628.
error(nargchk(2,6,nargin),'struct');
x_log = log10(x(:));
y_log = log10(y(:));
if length(x) ~= length(y)
    error('The number of elements in x and y do not match')
end
if nargin < 5
%     cmap = flipud(gray(256));
    cmap = jet;
end
if nargin < 4
    clm = 'k+';
end
if nargin < 3
    l = 1;
end    
if nargin < 2
    n = 25;
end
% min/max of x and y
minX = min(x_log);
maxX = max(x_log);
minY = min(y_log);
maxY = max(y_log);

% edge locations
xEdges = linspace(minX,maxX,n);
yEdges = linspace(minY,maxY,n);
% shift edges
xDiff = xEdges(2) - xEdges(1);
yDiff = yEdges(2) - yEdges(1);
xEdges = [-Inf, xEdges(2:end) - xDiff/2, Inf];
yEdges = [-Inf, yEdges(2:end) - yDiff/2, Inf];
% number of edges
numX = numel(xEdges);
numY = numel(yEdges);
% hold counts
C = zeros(numY,numX);
% do counts
for i = 1:numY-1
    for j = 1:numX-1
        C(i,j) = length(find(x_log >= xEdges(j) & x_log < xEdges(j+1) &...
                             y_log >= yEdges(i) & y_log < yEdges(i+1)));
    end
end
% get rid of Infs from the edges
xEdges = [xEdges(2) - xDiff,xEdges(2:end-1), xEdges(end-1) + xDiff];
yEdges = [yEdges(2) - yDiff,yEdges(2:end-1), yEdges(end-1) + yDiff];
% smooth the density data, in both directions.
C = localSmooth(localSmooth(C,l)',l)';
% create the graphics
% figure
% This section of plotting color map is commented out.
% ax = newplot;
% s = surf(xEdges,yEdges,zeros(numY,numX),C,...
%          'EdgeColor','none',...
%          'FaceColor','interp');
% view(ax,2);
% colormap(ax,cmap);
% colorbar
% grid(ax,'off');
% holdstate = get(ax,'NextPlot');
% set(ax,'NextPlot','add');
% p = plot(x,y,clm,'MarkerSize',3);
% axis(ax,'tight');
% set(ax,'NextPlot',holdstate)

% Add additional features on the plots
hold on
p = plot(x_log,y_log,clm,'MarkerSize',3);
% plot(minX:0.1:maxX,ones(1,length(minX:0.1:maxX)),'w--');
% plot(-1*ones(1,length(minY:0.1:maxY)),minY:0.1:maxY,'w--');
contour(xEdges,yEdges,C,5,clm,'LineWidth',1)
set(gca,'FontSize',15)
box on

% xlabel('log_{10}(Droplet mean pixel intensity)')
% ylabel('log_{10}(D_{eff-500ms} / 1\mum^2\cdots^{-1})')
% % ylabel('log_{10}(Deff / 1\mum^2\cdots^{-1})')

xlabel('log_{10}(Droplet GFP mean pixel intensity)')
ylabel('log_{10}(Droplet BFP mean pixel intensity)')

% outputs
if nargout
    h = p;
end
function B = localSmooth(A,L)
r = size(A,1);
I = eye(r);
D1 = diff(I);
D2 = diff(I,2);
B = (I + L ^ 2 * D2' * D2 + 2 * L * D1' * D1) \ A;
