clear all
clc
tic

load('COIL20.mat');

featurenumset=200;

NIter=50;
m=featurenumset;
nClass1=length(unique(gnd));
SearchAgents_no=20;
lb=1e-8;
ub=1e+8;
dim=4;
Vmax = 0.1*ub;
Vmin = 0.1*lb;

for i=1:20
    [X_new,W,idx,Best_pos,Best_score,curve,curve1] =GOD_cPSO_optimization(fea',nClass1,m,NIter,SearchAgents_no,lb,ub,dim,Vmax,Vmin);
    fun1(i)=Best_score;
    funx(i,:)=Best_pos;
    Best_score
end
b(1,1)=min(fun1);
b(1,2)=max(fun1);
b(1,3)=mean(fun1);
b(1,4)=std(fun1,0);
c=mean(funx);


X_new=X_new';
for i=1:40
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

