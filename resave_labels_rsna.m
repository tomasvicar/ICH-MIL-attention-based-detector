clc;clear all;close all;

tic

dcom_colection_name = '../rsna_dicomcolection/dicomColl.mat';
label_table_name = '../rsna_dicomcolection/stage_2_train.csv';

dcom_colection = load(dcom_colection_name).dicomColl_rsna_all;

label_table = readtable(label_table_name);



label_table_any = label_table(contains(label_table.ID,'_any'),:);

label_table_any.ID = cellfun(@(x) x(1:end-4), label_table_any.ID,'UniformOutput',0);

writetable(label_table_any,'../rsna_dicomcolection/label_table_any.csv','Delimiter',';');


MM = cellfun(@(x) size(x,1), dcom_colection{:,14},'UniformOutput',1);
M = sum(MM);

StudyInstanceUID = cell(M,1);
SeriesInstanceUID = cell(M,1);
Frames = zeros(M,1);
SliceNum = zeros(M,1);
PacNum = zeros(M,1);


ind = 1;
for k=1:size(dcom_colection,1)

    N = MM(k);
    
    
    StudyInstanceUID(ind:ind+N-1) = {dcom_colection{k,12}};
    SeriesInstanceUID(ind:ind+N-1) = {dcom_colection{k,13}};
    Frames(ind:ind+N-1) = dcom_colection{k,9};
    SliceNum(ind:ind+N-1) = 1:N;
    PacNum(ind:ind+N-1) = k;
    
    ind = ind+N;
    
end

ID = cat(1,dcom_colection{:,14}{:});

dicom_table = table(ID,StudyInstanceUID,SeriesInstanceUID,Frames,SliceNum,PacNum);



dicom_table.ID = cellfun(@(x) x(15:end-4), dicom_table.ID,'UniformOutput',0);




T = innerjoin(dicom_table,label_table_any,'Keys',{'ID','ID'});

writetable(T,'../rsna_dicomcolection/label_table_dicomcol_merge.csv','Delimiter',';');


