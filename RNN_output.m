function     [out,h]=RNN_output(V,W,input,len,h0,K)
out = zeros(len,1); 
h{1} = W{1}*h0;
for i = 2 : K
h{1} = h{1} + W{i}*h0*input(1)^(i-1);
end


out(1) =V*h{1};

for k=2 : len
    
    h{k} = W{1}*h{k-1};
    for i = 2 :K
        h{k} = h{k} + W{i}*h{k-1}*input(k)^(i-1);
    end
    out(k) = V*h{k};
end


end