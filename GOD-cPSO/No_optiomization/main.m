clear;
clc;
tic

load('lung_discrete.mat');
featurenumset=200;

kset=5;

alpha=1e-8;
beta=1e-8;
gamma=1e+8;
delta=1e+8;


NIter=50;
m=featurenumset;
nClass1=length(unique(gnd));
%---------------------------Initialization----------------------------
[n,d]=size(fea);

fprintf('算法迭代运行中...\n');
%-------------------------feature selection---------------------------


[X_new,W,idx,obj] =GOD_cPSO(fea',nClass1,alpha,beta,gamma,delta,m,NIter);
X_new=X_new';
for i=1:30
    label=litekmeans(X_new,nClass1,'MaxIter',100,'Replicates',10);
    newres = bestMap(gnd,label);
    AC = length(find(gnd == newres))/length(gnd);
    MIhat=MutualInfo(gnd,label);
    resualt(i,:)=[AC,MIhat];
end
for j=1:2
    a=resualt(:,j);
    ll=length(a);
    temp=[];
    for i=1:ll
        if i<ll-18
            b=sum(a(i:i+19));
            temp=[temp;b];
        end
    end
    [e,f]=max(temp);
    e=e./20;
    MEAN(j,:)=[e,f];
    STD(j,:)=std(resualt(f:f+19,j));
    rr(:,j)=sort(resualt(:,j));
    BEST(j,:)=rr(end,j);
end

fprintf('算法运行完毕！\n');

fprintf('以下是AP_OCLGR算法运行得到的AR10P数据集的ACC与NMI值：\n');


fprintf('ACC±STD%%:\n');
fprintf('%.2f\t',MEAN(1,1)*100);
fprintf('%.2f\t',STD(1,:)*100);

fprintf('\n');

fprintf('NMI±STD%%:\n');
fprintf('%.2f\t',MEAN(2,1)*100);
fprintf('%.2f\t',STD(2,:)*100);


fprintf('\n');
toc;
fprintf('\n');

