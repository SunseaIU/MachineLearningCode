% 核心函数：
function [final_U,final_V,final_history_obj] = gflcf(X,c,lambda,gamma,options)
%X: d*n, must be nonnegtative
%c: number of cluster
%lamda,gamma: parameter
%optins:
%   maxiter: max times of each iterative
%   trynum: different initialization of U,V, try times of main iterative
%   mu,beta: ALM parameter;
if min(min(X))<0
    error('matrix entries can not be negative');
    return;
end
if min(sum(X,2)) == 0
    zeroidx = find(sum(X,2) == 0);
    X(zeroidx, :) = [];
    fprintf('not all entries in a row can be zero');
end

[d,n]=size(X);
maxiter=options.maxiter;
trynum=options.trynum;
mu=options.mu;
beta=options.beta;
%calculate graph Laplacian matrix L
options1=[];
options1.NeighborMode = 'KNN';
options1.k = 5;
options1.WeightMode = 'HeatKernel';
options1.t = 1;
W = constructW(X',options1);
W=full(W);
D=diag(sum(W,2));
L=D-W;
%end calculate L
for tn=1:trynum
   %Initialize random U,V
   U=rand(d,c);
   V=rand(n,c);
   E=zeros(n,c);
   history_obj=[];
   %main iteration
    for i=1:maxiter
       %update U
       H=diag(sum(V)); %c*c
       U=U.*(((lambda+1)*X*V)./(U*(V'*V)+lambda*U*H+1e-9));
       %update V
       for a=1:n
           for k=1:c
                tmp=X(:,a)-U(:,k);
                E(a,k)=0.5*norm(tmp)^2;
           end
       end
       clear a;
       clear k;
       clear tmp;
       tmp=U'*X-lambda*E';
       for a=1:n
           A=U'*U+gamma*D(a,a)*eye(c);
           b=[];
           gsum=zeros(1,c);
           for g=1:n
               gsum=gsum+W(a,g)*V(g,:);
           end
           b=2*(tmp(:,a)+gamma*gsum');
           x=SimplexQP_ALM(A,b,mu,beta,0);
           V(a,:)=x;
       end
       %calculate obj
       obj=CalculateObj(X,U,V,lambda,L,gamma);
       history_obj=[history_obj;obj];
    end
    %judge optimun solution
    if tn==1
        final_history_obj=history_obj;
        final_U=U;
        final_V=V;
        final_obj=obj;
    else
        if obj< final_obj
            final_history_obj=history_obj;
            final_U=U;
            final_V=V;
            final_obj=obj;
        end
    end
end
end

% 可视化模糊聚类算法的结果，并标记模糊的数据点：

clear();
clc();

调整的参数
model='GFLCF';
fontsize=14;

% 每个聚类的颜色和形状
markerStyles = {'b*','c+','md','gs','ro'}; % b*:蓝星, c+:青十字, m:菱形, g:方形, r:圆圈
%END

load('result.mat');
[~,label]=max(V,[],2);
label=bestMap(gnd,label);

% 创建图形并设置背景色为白色
figure('Color', 'white'); % 设置图形窗口背景为白色
hold on;

numClusters = length(unique(label));

% 绘制各簇数据
for k = 1:numClusters
    idx = find(label == k);
    plot(fea(idx,1), fea(idx,2), markerStyles{k}, 'MarkerSize', 8);
end

% 绘制模糊点
for index = 1:length(fuzzypoint)
    i = fuzzypoint(index);
    plot(fea(i,1), fea(i,2),'ko','MarkerSize',10,'LineWidth',1.5);
    text(fea(i,1)+0.05, fea(i,2)-0.01, num2str(i),'fontsize',16);
end

set(gca,'fontsize',14);
xlabel('特征 1','fontsize',14);
ylabel('特征 2','fontsize',14);


% 添加图例
legendStrings = arrayfun(@(x) sprintf('类别 %d', x), 1:numClusters, 'UniformOutput', false);
legend(legendStrings,'Location','best');

print -depsc fea_f.eps
print -dtiff -r500 fea_f.tiff

% 展示模糊聚类的结果，通过将模糊聚类的成员度矩阵可视化，来识别和展示数据点的隶属度：
clear();
clear;
clc;

%调整的参数
model='GFLCF';
xtick=[0,100,200,300,400,500];
ytick=[0,100,200,300,400,500];
defaultLineWidth=1.5;        % 默认线宽
blueLineWidth=1;             % 蓝色线宽
markerSize=6;                % 标记大小
rownum=5;                     % 图的行数
columnnum=1;                  % 图的列数
Vrange=[1,201,276,351,426,501];
Vcluster=[5,4,3,1,2];        % 聚类标签顺序
dottedlinecolor='k';          % 虚线颜色
%END

load('result.mat');

fuzzypoint=[]; % 记录模糊点坐标
index=randperm(size(fea,1));
index=sort(index);

% 为黑白打印准备线型+标记组合
lineStyles = {'-o','--s',':d','-.^','-+'};

figure('Color','white');

% 竖向布局
if columnnum==1
    for row=1:rownum
        pos=row;
        subplot(rownum,columnnum,pos);
        yt=[];

        % 判断是否为蓝色簇（假设蓝色簇为第1个）
        if pos==1
            lw = blueLineWidth;
        else
            lw = defaultLineWidth;
        end

        plot(index, V(:,Vcluster(pos)), lineStyles{pos}, 'LineWidth', lw, 'MarkerSize', markerSize);
        axis([-inf,inf,0,1]);

        % 处理模糊点
        for i=1:size(V,1)
            val = V(i,Vcluster(pos));
            if (i>=Vrange(pos) && i<Vrange(pos+1) && val~=1) || ...
               (i<Vrange(pos) && val~=0) || (i>=Vrange(pos+1) && val~=0)
                line([i,i],[0,val],'LineStyle',':','Color',dottedlinecolor,'LineWidth',1);
                yt=[yt,i];
                fuzzypoint=[fuzzypoint,i];
            end
        end

        ytick1 = sort([ytick, yt]);
        set(gca,'YTick',ytick1,'YTickLabel',ytick1,'FontSize',12);
        grid on;
        ax=gca;
        ax.GridColor=[0.8 0.8 0.8];
        ax.GridAlpha=0.3;
    end
end

fuzzypoint=sort(fuzzypoint);