load 'CNNparameters.mat'
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

%loading this file defines imageset, trueclass, and classlabels
load 'cifar10testdata.mat'
%some sample code to read and display one image from each class
% for classindex = 1:10
%     %get indices of all images of that class
%     inds = find(trueclass==classindex);
%     %take first one
%     imrgb = imageset(:,:,:,inds(1));
%     %display it along with ground truth text label
%     figure; imagesc(imrgb); truesize(gcf,[64 64]);
%     title(sprintf('%s', classlabels{classindex}));
% end

% runs the first image of the dataset through the imnormalize and apply
% relu layers
imrgb = imageset(:,:,:,1);
figure;imagesc(imrgb);truesize(gcf,[64,64]);
outarray = apply_imnormalize (imageset(:,:,:,1));
figure;imagesc(outarray);truesize(gcf,[64,64]);
outarray = apply_relu (outarray);
figure;imagesc(outarray);truesize(gcf,[64,64]);



%loading this file defines imrgb and layerResults
load 'debuggingTest.mat'
%sample code to show image and access expected results
% figure; imagesc(outarray); truesize(gcf,[64 64]);
% for d = 1:length(layerResults)
%     result = layerResults{d};
%     fprintf('layer %d output is size %d x %d x %d\n',...
%     d,size(result,1), size(result,2), size(result,3));
% end
% %find most probable class
% classprobvec = squeeze(layerResults{end});
% [maxprob,maxclass] = max(classprobvec);
% %note, classlabels is defined in 'cifar10testdat.mat'
% fprintf('enstimated class is %s with probability %.4f\n',...
% classlabels{maxclass},maxprob);

% function definitions for apply normalize and relu
function outarray = apply_imnormalize (inarray)
    [n,m,d1] = size(inarray);
    outarray = zeros(n,m,d1);
    for k = 1:d1
        for j = 1:m
            for i = 1:n
                outarray(i,j,k) = inarray(i,j,k)/255-0.5;
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
                outarray(i,j,k) = max(inarray(i,j,k),0);
            end
        end
    end
end

function outarray = apply_maxpool(inarray)
    
end
