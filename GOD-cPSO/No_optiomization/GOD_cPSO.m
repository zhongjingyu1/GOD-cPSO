function [X_new,W,idx,obj] = GOD_cPSO(X,n_class,alpha,beta,gamma,delta,m,NIter)


[nFeat,nSamp] = size(X);%1024  1440

W=ones(nFeat,n_class);
B=rand(n_class,n_class);
E=rand(nSamp,n_class);
Z=rand(nSamp,n_class);

%% 双图初始化

options = [];
options.NeighborMode = 'KNN';
options.k =5;
options.t =1e+4;
options.WeightMode = 'Heatkernel';

S1=constructW(X',options);
Ds=diag(sum(S1));
Ls=Ds-S1;

S2=constructW(X,options);
D2=diag(sum(S2));
Lw=D2-S2;


%% 优化过程
for i=1:NIter

    Wi = sqrt(sum(W.*W,2)+eps) ;
    d = 0.5./Wi;
    Da = diag(d);

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



    obj(i)=norm((W'*X-B*E'),'fro')^2+alpha*trace(W'*Lw*W)+beta*(trace(W'*Da*W)-norm(W,2))+gamma*(trace(W'*X*Ls*X'*W))+delta*norm((E-Z),'fro')^2;
    %  fprintf('%f\n',obj);

end

score= sqrt(sum(W.*W,2));
[res, idx] = sort(score,'descend');
X_new = X(idx(1:m),:);
end