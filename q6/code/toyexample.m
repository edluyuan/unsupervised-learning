% run both standard and collapsed gibbs on the toy example consisting of
% 6 documents, 6 words, and 3 topics.  The true word distribution for each
% topic should be:
% [.5 .5 0 0 0 0], [0 0 .5 .5 0 0], [0 0 0 0 .5 .5]

K = 3;             % number of topics
alpha = 1;         % dirichlet prior over topics
beta =  1;         % dirichlet prior over words
numiter = 200;     % number of iterations

[I,D,K,W,di,wi,ci,citest,Id,Iw,Nd] = lda_read('toyexample.data',K);

[zi,theta,phi] = lda_randstate(I,D,K,W,di,wi,ci,citest,Id,Iw,Nd,alpha,beta);

[zistdgibbs theta phi Adk Bkw Mk Lstdgibbs Pstdgibbs Tstdgibbs] ...
        = stdgibbs_run(zi,theta,phi,numiter,...
        I,D,K,W,di,wi,ci,citest,Id,Iw,Nd,alpha,beta);

[zicolgibbs Adk Bkw Mk Lcolgibbs Pcolgibbs Tcolgibbs] ...
        = colgibbs_run(zi,numiter,...
        I,D,K,W,di,wi,ci,citest,Id,Iw,Nd,alpha,beta);

subplot(221); plot(1:201,Pstdgibbs); title('std gibbs log pred');
subplot(222); plot(1:201,Lstdgibbs); title('std gibbs log joint');
subplot(223); plot(1:201,Pcolgibbs); title('col gibbs log pred');
subplot(224); plot(1:201,Lcolgibbs); title('col gibbs log joint');


