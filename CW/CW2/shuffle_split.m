load('data')
clc
s = RandStream('mt19937ar','Seed',1);       % Used for training with same shuffling
RandStream.setGlobalStream(s);
shuffled = randperm(s,size(data,1))';
data = data(shuffled,:);

train_data = data(5000:end,:);
test_data = data(1,:);


save('train_data.mat','train_data')
save('test_data.mat','test_data')
