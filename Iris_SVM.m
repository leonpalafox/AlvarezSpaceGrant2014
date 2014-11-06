%This function will read in data from the UC Irvine Iris data set and will
%utilize a one-vs.-all approach to classify the data using the Matlab SVM
%library.

%Data comes from the UC Irvine repository found in this file
[Num, Txt, Raw]=xlsread('/Users/kenpo_stud/Documents/MATLAB/SpaceGrant/Iris_Data.xlsx');

%Define the x as a matrix of observations where the rows indicate an
%observation and the column represents a different classifier
X=Num;
[M, N]=size(X);
%Define Y as the text outputs and place them in the categorical array
Y=Txt;

%To do a one vs. all approach, call all Iris-setosa as 1 and all other
%labels as 0
Y_log = strcmp('Iris-setosa',Y);
Y_log = Y_log-1;
Y_log(Y_log==0)=1;
for i=1:3
    if i==1
        Kernel_Func='linear';
    elseif i==2
        Kernel_Func='rbf';
    elseif i==3
        Kernel_Func='polynomial';
    end
    
    %Matlab utilizes the fitcsvm function to classify data. The entire data
    %set, X, is used as the training set (because it is a small data set
    %and cross-validation can be used to train correctly and test
    %appropriately for each fold). Y_log is used as the labels. Then the
    %following options are used. 'KernelFunction' calls the three different
    %types of kernels used in the function as defined by the for loop
    %('linear', 'rbf' for Gaussian, and 'polynomial' of order 3);
    %'Standardize' normalizes all tof the parameters so that specifically
    %larger parameters are not overweighted. 
    Class_Data=fitcsvm(X,Y_log,'KernelFunction',Kernel_Func,'Standardize',true, 'BoxConstraint', 1);
    %Cross validate the data and find the parameter for the box constraint
    %that minimizes the kFoldLoss
    Cross_Class_Data=crossval(Class_Data);
    %Define minfn as the function that minimizes the kfoldLoss by modfiying
    %the box constraint parameter
    minfn=@(z)kfoldLoss(fitcsvm(X,Y_log,'KernelFunction',Kernel_Func,'Standardize',true, 'BoxConstraint', z(1), 'KernelScale', z(2) ));
    %Loosen the tolerance for use in the function
    opts=optimset('TolX',5e-4,'TolFun',5e-4);
    %By changing the 
    m=20;
    fval=zeros(m,1);
    z=ones(m,2);
    for j=1:m
        [searchmin, fval(j)]=fminsearch(minfn,randn(2,1),opts);
        z(j,:)=exp(searchmin);
    end
    z=z(fval == min(fval),:);
    Class_Data=fitcsvm(X,Y_log,'KernelFunction',Kernel_Func,'Standardize',true, 'BoxConstraint', z(1),'KernelScale',z(2));
    [~, Y_predict(:,i)]=predict(Class_Data,X);
    Class_Data_Posterior=fitSVMPosterior(Class_Data);
    [~, Post_predict(:,i)]=predict(Class_Data_Posterior,X);
    %Initialize elements of confusion matrix and ROC curve
    Num_TP=0; Num_TN=0; Num_FP=0; Num_FN=0;

    for j=1:length(Y_log)
        %Define a True Positive (TP) as a setos that is classified as a
        %setosa
        if Y_predict(j,i)==1 && Y_log(j)==1
            Num_TP=Num_TP+1;
        %Define a True Negative (TN) as other irises that are classified
        %as other irises
        elseif Y_predict(j,i)==0 && Y_log(j)==0
            Num_TN=Num_TN+1;
        %Define a False Positive (FP) as another iris that is classified as a
        %setosa
        elseif Y_predict(j,i)==1 && Y_log(j)==0
            Num_FP=Num_FP+1;
        %Define a False Negative (FN) as a setosa that is classified as
        %another iris
        elseif Y_predict(j,i)==0 && Y_log(j)==1
            Num_FN=Num_FN+1;
        end

    end
    %Define sensitivity
    Se(i)=Num_TP/(Num_TP+Num_FN);
    %Define specificity
    Sp(i)=Num_TN/(Num_TN+Num_FP);
    %Define the false positive rate as 1.-Sp
    FP_rate(i)=1.-Sp(i);

    %Plot an ROC curve and label it in the legend 
    %Transpose for ease of use
    Y_working=Y_predict(:,i)';
    Y_working2=Post_predict(:,i)';
    %targets is a 2x150 matrix that represents a 1 in the row of whichever
    %category the number represents and a 0 in the other row
    targets(1,Y_working==1)=1;
    targets(2,Y_working==1)=0;
    targets(1,Y_working==-1)=0;
    targets(2,Y_working==-1)=1;
    %outputs is a 2x150 matrix that represents the posterior probability
    %for each category
    outputs(1,Y_working2==1)=1;
    outputs(2,Y_working2==1)=0;
    outputs(1,Y_working2==-1)=0;
    outputs(2,Y_working2==-1)=1;
    [tpr(i), fpr(i), thresholds(i)]=roc(targets, outputs);
    plotroc
    %Define a 2x2 confusion matrix that compares the number of TP, TN, FP, and
    %FN
    Conf_matrix(:,:,i)=[Num_TP, Num_FN; Num_FP, Num_TN];

end
