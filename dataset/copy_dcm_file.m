%% 

clear all
close all
clc

% addpath('export_fig\')

path_save = ['D:\Jakubicek\RSNA\data\RSNA_sub'];
mkdir(path_save)

path = 'D:\Jakubicek\RSNA\data\stage_2_train';
% path = 'D:\Jakubicek\RSNA\data\RSNA_sub';

anot = readtable('D:\Jakubicek\RSNA\data\stage_2_train.csv');
anotU = anot(6:6:end,:);


   %%
tic
parfor i = [1:size(anotU,1)]
% i
%% data loading
    img = dicomread([path '/' anotU.ID{i}(1:end-4) '.dcm' ]);
    img = imresize(img,0.5,'AntiAliasing',true,'Method','nearest');
    dicomwrite(img,[path_save '\' anotU.ID{i}(1:end-4) '.dcm'])
    

end
toc
