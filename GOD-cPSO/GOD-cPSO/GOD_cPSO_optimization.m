function [X_new1,best_W,idx,Best_pos,Best_score,curve,curve1] = GOD_cPSO_optimization(X1,n_class,m,NIter,sizepop,lb,ub,Dim,Vmax,Vmin)

% X1=fea';
% n_class=7;
% sizepop=16;
% Dim=4;
wmax = 0.9;      % 惯性因子
wmin = 0.1;
c1 = 2;       % 加速常数
c2 = 2;       % 加速常数

%% 粒子群初始化

[nFeat,nSamp] = size(X1);%1024  1440

best_W=ones(nFeat,n_class);
best_B=rand(n_class,n_class);
best_E=rand(nSamp,n_class);
best_Z=rand(nSamp,n_class);

%% 双图初始化

options = [];
options.NeighborMode = 'KNN';
options.k =5;
options.t =1e+4;
options.WeightMode = 'Heatkernel';

S1=constructW(X1',options);
Ds=diag(sum(S1));
Ls=Ds-S1;

S2=constructW(X1,options);
D2=diag(sum(S2));
Lw=D2-S2;

Wi = sqrt(sum(best_W.*best_W,2)+eps) ;
d = 0.5./Wi;
Da = diag(d);



if(max(size(ub)) == 1)
    ub = ub.*ones(1,Dim);
    lb = lb.*ones(1,Dim);
end

Range = ones(sizepop,1)*(ub-lb);
pop = rand(sizepop,Dim).*Range + ones(sizepop,1)*lb;    % 初始化粒子群

% pop(1,:)=[1e+8,1e+8,1e+8,1e+8];
% pop(2,:)=[1e+8,1e+8,1e+8,1e-8];
% pop(3,:)=[1e+8,1e+8,1e-8,1e+8];
% pop(4,:)=[1e+8,1e+8,1e-8,1e-8];
% pop(5,:)=[1e+8,1e-8,1e+8,1e+8];
% pop(6,:)=[1e+8,1e-8,1e+8,1e-8];
% pop(7,:)=[1e+8,1e-8,1e-8,1e+8];
% pop(8,:)=[1e+8,1e-8,1e-8,1e-8];
% pop(9,:)=[1e-8,1e+8,1e+8,1e+8];
% pop(10,:)=[1e-8,1e+8,1e+8,1e-8];
% pop(11,:)=[1e-8,1e+8,1e-8,1e+8];
% pop(12,:)=[1e-8,1e+8,1e-8,1e-8];
% pop(13,:)=[1e-8,1e-8,1e+8,1e+8];
% pop(14,:)=[1e-8,1e-8,1e+8,1e-8];
% pop(15,:)=[1e-8,1e-8,1e-8,1e+8];
% pop(16,:)=[1e-8,1e-8,1e-8,1e-8];



V = rand(sizepop,Dim).*(Vmax-Vmin) + Vmin;                 % 初始化速度

fitness = zeros(sizepop,1);
for i=1:sizepop
    [fitness(i),Wa(:,:,i),Ea(:,:,i),Ba(:,:,i),Za(:,:,i)]= optimization1(best_W,X1,best_E,best_B,best_Z,Da,Ls,Lw,n_class,pop(i,1),pop(i,2),pop(i,3),pop(i,4));
end

[bestf, bestindex]=min(fitness);
zbest=pop(bestindex,:);   % 全局最佳
gbest=pop;                % 个体最佳
fitnessgbest=fitness;              % 个体最佳适应值
fitnesszbest=bestf;               % 全局最佳适应值

best_W=Wa(:,:,bestindex);
best_E=Ea(:,:,bestindex);
best_B=Ba(:,:,bestindex);
best_Z=Za(:,:,bestindex);

%% 优化过程
for iter=1:NIter

    Wi = sqrt(sum(best_W.*best_W,2)+eps) ;
    d = 0.5./Wi;
    Da = diag(d);
    w=wmax-(wmax-wmin)*iter/NIter;
    for j=1:sizepop
        % 速度更新
        V(j,:) = w*V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        if V(j,:)>Vmax
            V(j,:)=Vmax;
        end
        if V(j,:)<Vmin
            V(j,:)=Vmin;
        end
        % 位置更新
        pop(j,:)=pop(j,:)+V(j,:);
        for k=1:Dim
            if pop(j,k)>ub(k)
                pop(j,k)=ub(k);
            end
            if pop(j,k)<lb(k)
                pop(j,k)=lb(k);
            end
        end
        % 适应值
        [fitness(j),Wa(:,:,j),Ea(:,:,j),Ba(:,:,j),Za(:,:,j)]=optimization1(best_W,X1,best_E,best_B,best_Z,Da,Ls,Lw,n_class,pop(j,1),pop(j,2),pop(j,3),pop(j,4));

        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        % 群体最优更新
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
            best_W=Wa(:,:,j);
            best_E=Ea(:,:,j);
            best_B=Ba(:,:,j);
            best_Z=Za(:,:,j);
        end
    end
    curve(iter) = fitnesszbest;
    curve1(iter,:)=zbest;
end
Best_pos = zbest;
Best_score = fitnesszbest;

score= sqrt(sum(best_W.*best_W,2));

[res, idx] = sort(score,'descend');
X_new1 = X1(idx(1:m),:);
end


function [obj,W,E,B,Z]=optimization1(W,X,E,B,Z,Da,Ls,Lw,n_class,alpha,beta,gamma,delta)

[nFeat,nSamp] = size(X);%1024  1440
W=W.*((X*E*B'+0.5*beta*((trace(W*W'))^(-0.5))*W)./((X*X'*W)+beta*Da*W+gamma*X*Ls*X'*W+alpha*Lw*W+eps));

T1=E'*X'*W;
[UB,a1,VB] = svd(T1);
B=VB*eye(n_class,n_class)*UB';

AA=alpha*Ls;
B1=2*X'*W*B+2*delta*Z;
[x1,y1]=eigs(AA);
m1=diag(y1);
u1=max(m1);
AA1=u1*eye(nSamp)-AA;
P=AA1*E+B1/2;
[Um,a1,Vm] = svd(P);
E=Um*eye(nSamp,n_class)*Vm';

Z=max(E,0);

obj=norm((W'*X-B*E'),'fro')^2+alpha*trace(W'*Lw*W)+beta*(trace(W'*Da*W)-norm(W,2))+gamma*(trace(W'*X*Ls*X'*W))+delta*norm((E-Z),'fro')^2;
obj=abs(obj);
end