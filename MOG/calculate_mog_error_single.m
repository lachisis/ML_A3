function pred = calculate_mog_error_single(data,p1,mu1,vary1,...
                                    p2,mu2,vary2,...
                                    p3,mu3,vary3,...
                                    p4,mu4,vary4,...
                                    p5,mu5,vary5,...
                                    p6,mu6,vary6,...
                                    p7,mu7,vary7)
                                %returs the classification
 train = [mogLogProb(p1,mu1,vary1,data);...
                mogLogProb(p2,mu2,vary2,data);...
                mogLogProb(p3,mu3,vary3,data);...
                mogLogProb(p4,mu4,vary4,data);...
                mogLogProb(p5,mu5,vary5,data);...
                mogLogProb(p6,mu6,vary6,data);...
                mogLogProb(p7,mu7,vary7,data)];
[maxval,pred] = max(train);                             
end                                