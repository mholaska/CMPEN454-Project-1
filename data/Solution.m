load 'CNNparameters.mat'
load 'cifar10testdata.mat'

for d = 1:length(layertypes)
    fprintf('layer %d is of type %s\n',d,layertypes{d});
    filterbank = filterbanks{d};
    if not(isempty(filterbank))
        fprintf(' filterbank size %d x %d x %d x %d\n', ...
        size(filterbank, 1),size(filterbank,2), ...
        size(filterbank,3),size(filterbank,4));
        biasvec = biasvectors{d};
        fprintf(' number of biases is %d\n',length(biasvec));
    end
end


confusion_matrix = zeros(10,10);
output = zeros(1,1,10,10000);

for img_index = 1:10000
    imrgb = imageset(:,:,:,img_index);
    output(:,:,:,img_index) = run_CNN(imrgb, filterbanks, biasvectors);
    [probability, index] = max(output(:,:,:,img_index));
    confusion_matrix(trueclass(img_index),index) = confusion_matrix(trueclass(img_index),index) + 1;
end

function outarray = run_CNN(inarray, filterbanks, biasvectors)
    outarray = apply_imnormalize (inarray);
    outarray = apply_convolve (outarray, filterbanks{2}, biasvectors{2});
    outarray = apply_relu(outarray);
    outarray = apply_convolve (outarray, filterbanks{4}, biasvectors{4});
    outarray = apply_relu(outarray);
    outarray = apply_maxpool(outarray);
    outarray = apply_convolve (outarray, filterbanks{7}, biasvectors{7});
    outarray = apply_relu(outarray);
    outarray = apply_convolve (outarray, filterbanks{9}, biasvectors{9});
    outarray = apply_relu(outarray);
    outarray = apply_maxpool(outarray);
    outarray = apply_convolve (outarray, filterbanks{12}, biasvectors{12});
    outarray = apply_relu(outarray);
    outarray = apply_convolve (outarray, filterbanks{14}, biasvectors{14});
    outarray = apply_relu(outarray);
    outarray = apply_maxpool(outarray);
    outarray = apply_fullconnect(outarray, filterbanks{17}, biasvectors{17});
    outarray = apply_softmax(outarray);
end

function outarray = apply_imnormalize (inarray)
    [n,m,d1] = size(inarray);
    outarray = zeros(n,m,d1);
    for k = 1:d1
        for j = 1:m
            for i = 1:n
                outarray(i,j,k) = (inarray(i,j,k)/255.0)-0.5;
            end
        end
    end
end

function outarray = apply_relu(inarray)
    [n,m,d1] = size(inarray);
    outarray = zeros(n,m,d1);
    for k = 1:d1
        for j = 1:m
            for i = 1:n
                if(inarray(i,j,k) < 0)
                    outarray(i,j,k) = 0;
                else
                    outarray(i,j,k) = inarray(i,j,k);
                end
                
            end
        end
    end
end

function outarray = apply_maxpool(inarray)
    [n,m,d1] = size(inarray);
    outarray = zeros(n/2,m/2,d1);
    for k = 1:d1
        for j = 1:(m/2)
            for i = 1:(n/2)
                array_section = inarray((2*i)-1:2*i,(2*j)-1:2*j,k);
                outarray(i,j,k) = max(array_section,[],"all");
            end
        end
    end
end

function outarray = apply_convolve(inarray, filterbank, biasvals)
    [n,m,d1] = size(inarray);
    d2 = size(filterbank,4);
    outarray = zeros(n,m,d2);
    for i = 1:d2
        total = 0;
        for j = 1:d1
            total = total + convn(inarray(:,:,j),filterbank(:,:,j,i),'same') ;
        end
        outarray(:,:,i) = total+ biasvals(i);
    end
end

function outarray = apply_fullconnect(inarray, filterbank, biasvals)
    [n,m,d1] = size(inarray);
    d2 = size(filterbank,4);
    outarray = zeros(1,1,d2);
    for l = 1:d2
        total = 0;
        for k = 1:d1
            for j = 1:m
                for i = 1:n
                    total = total + (filterbank(i,j,k,l)*inarray(i,j,k));
                end
            end
        end
        outarray(:,:,l) = total+biasvals(l);
    end
end

function outarray = apply_softmax(inarray)
    [n,m,d] = size(inarray);
    outarray = zeros(n,m,d);
    denom = zeros(n,m,d);
    alpha = max(inarray);
    for i = 1:d
        denom(i) = exp(inarray(:,:,i)-alpha);
    end
    for i = 1:d
        outarray(:,:,i) = exp(inarray(:,:,i)-alpha)/sum(denom);
    end
    
end