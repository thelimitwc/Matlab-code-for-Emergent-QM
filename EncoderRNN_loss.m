function loss = EncoderRNN_loss(Y,W1,b1,W2,b2,WR,v,K,len,n,m,N)
loss =0;
for k= 1 :len
    h{k} = zeros(n,1); z{k}= zeros(m,1);s{k}=zeros(N,1);
    y{k}=zeros(n,1); 
end

z{1} = ones(m,1); 
for k = 2 :len
    for i = 1 : K
        z{k}= z{k} + WR{i}*z{k-1}*v(k-1)^(i-1);
    end
end
  %H = zeros(len*n,1);
for k = 1:len 
    y{k} = Y(1+(k-1)*n : k*n);
    s{k}=Ramp(W1*z{k}+b1);
    h{k}=W2*s{k}+b2;
    %H(1+(k-1)*n : k*n) = h{k};
    loss = loss + sum((h{k}-y{k}).^2);
end
%plot(H,Y,'.');
