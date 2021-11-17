%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                     The evaluation code for 
%%% SHREC'12 -- Shape Retrieval Contest based on Generic 3D Dataset  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input:
%       classification file  -- "SHREC2012_Generic_labels.cla"
%       distance matrix      -- "result.matrix"

%Output:
%       distance matrix for PSB evaluation code -- "PSBresult.matrix"
%       evaluation file      -- "result.txt"

%Evaluation measures:
%       NN, 1-Tier, 2-Tier, e_Measure, DCG

%Author: Bo Li
%Email: libo0002@ntu.edu.sg
%Date: February 24, 2012
%@NIST, Gaithersburg, US

%Please cite:
% SHREC'12 -- Shape Retrieval Contest based on Generic 3D Dataset, 3DOR'12, 2012


%---Please change the name of the folder if necessary!!
%---Please assign the name of the distance matrix (disFileName) in Line 106!!
%---Please use the classification file to "SHREC2012_Generic_labels.cla"!!

clear;
%The folder that contains the distance matrix and classification file,
filePath = '/home/apoorva/Apoorva_Minor/PerformanceEvaluation_complete/';

%Initialization
avgFirst_NN = 0;
first_NN = 0;
avgFirst_Tier = 0;
first_Tier = 0;
avgSecond_Tier = 0;
second_Tier = 0;
avgE_Measure = 0;
e_Measure = 0;
idealDCG = 0;
DCG = 0;
avgDCG = 0;
K1 = 0;
K2 = 0;

testCategoryList.categories(1).name(1) = 0;
testCategoryList.categories(1).numModels = 0;

testCategoryList.numCategories = 0;
testCategoryList.numTotalModels = 0;
testCategoryList.modelsNo(1) = 0;
testCategoryList.classNo(1) = 0;

%%%%%%Read the classification file
claFileName = sprintf('%stest_Shrec12.cla',filePath);
fp = fopen(claFileName,'r');

%Check file header
disp(claFileName)
strTemp = fscanf(fp,'%s',1);
if ~strcmp(strTemp,'PSB')
    display('The format of your classification file is incorrect!');
    return;
end
strTemp = fscanf(fp,'%s',1);
if ~strcmp(strTemp,'1')
    display('The format of your classification file is incorrect!');
    return;
end

numCategories = fscanf(fp,'%d',1);
numTotalModels = fscanf(fp,'%d',1);

testCategoryList.numCategories = numCategories;
testCategoryList.numTotalModels = numTotalModels;

currNumCategories = 0;
currNumTotalModels = 0;

for i=1:numCategories
    currNumCategories = i;
    testCategoryList.categories(currNumCategories).name = fscanf(fp,'%s',1);
    fscanf(fp,'%d',1);
    numModels = fscanf(fp,'%d\n',1);
    testCategoryList.categories(currNumCategories).numModels = numModels;
    for j=1:numModels
        currNumTotalModels = currNumTotalModels+1;
        TLine=fgets(fp);
        index=str2num(TLine(1,3:6));
        testCategoryList.modelsNo(currNumTotalModels) = index;
        testCategoryList.classNo(currNumTotalModels) = currNumCategories;
    end
end
disp(currNumTotalModels)
disp(numTotalModels)
if (currNumTotalModels~=numTotalModels)
    display('The format of your classification file is incorrect!');
    return;
else
    display('The format of your classification file is correct!');
end
fclose(fp);

%%%%%%Read the distance matrix, change the name of the distance matrix file to yours 
disFileName = sprintf('%sdistance.txt',filePath);
fp = fopen(disFileName,'r');
matrixInput = fscanf(fp,'%f');
numElement = size(matrixInput,1);
if (numElement~=(numTotalModels*numTotalModels))
    display('The format of your distance file is incorrect!');
    return;
else
    display('The format of your distance file is correct!');    
end
fclose(fp);


%%%%%%%Output the new distance matrix that can be used by the PSB evaluation code
disNewFileName = sprintf('%sPSBresult.matrix',filePath);
fp = fopen(disNewFileName,'w');
matrixDis(numTotalModels,numTotalModels) = 0;
for i=1:numTotalModels
    for j=1:numTotalModels
        iNew = testCategoryList.modelsNo(i);
        jNew = testCategoryList.modelsNo(j);
        matrixDis(i,j) = matrixInput((iNew-1)*numTotalModels+jNew);
        fprintf(fp,'%.6f ',matrixDis(i,j));             
    end
    fprintf(fp,'\n');
end

fclose(fp);

%%%%%%%Evaluation
matrixNo(1:numTotalModels) = 0;
modelNo(1:numTotalModels) = 0;
tempDis(1:numTotalModels) = 0;
for i = 1:numTotalModels
    matrixDis(i,i) = -Inf;
    [tempDis, modelNo] = sort(matrixDis(i,:));
    for k = 1:numTotalModels
        matrixNo(k) = testCategoryList.classNo(modelNo(k));
    end
	
	count = 0;
	K1 = testCategoryList.categories(matrixNo(1)).numModels-1;
	K2 = 2*K1;
	DCG = 0;
	idealDCG = 1;
	for j = 2:K1
		idealDCG = idealDCG + log(2.0)/log(j);
	end

	for j = 1:numTotalModels			
		if (matrixNo(j) == testCategoryList.classNo(i))
			count = count+1;
			if (j ~= 1)
				if (j == 2)
					first_NN = first_NN+1;
					DCG = 1;
				else
					DCG = DCG + log(2.0)/log(j-1);
				end
			end
		end
		if (j == K1+1)
			first_Tier = (count-1)*1.0/K1;
			avgFirst_Tier = avgFirst_Tier + first_Tier;
		end
		if (j == K2+1)
			second_Tier = (count-1)*1.0/K1;
			avgSecond_Tier = avgSecond_Tier + second_Tier;
		end
		
		if (j == 33)
			e_Measure = (count-1)*2.0/(K1+32);
			avgE_Measure = avgE_Measure + e_Measure;
		end
	end
	DCG = DCG/idealDCG;
	avgDCG = avgDCG + DCG;
	
end

avgFirst_Tier = avgFirst_Tier/numTotalModels;
avgSecond_Tier = avgSecond_Tier/numTotalModels;
avgE_Measure = avgE_Measure/numTotalModels;
avgDCG = avgDCG/numTotalModels;
avgFirst_NN = first_NN/numTotalModels;



fileName = argv(){1};
fp = fopen(fileName, 'a');

fprintf(fp, 'NN      1-Tier      2-Tier     e-Measure     DCG\n');
fprintf(fp, '%.4f  %.4f      %.4f     %.4f        %.4f\n', avgFirst_NN, avgFirst_Tier, avgSecond_Tier, avgE_Measure, avgDCG);

disp(' ')
disp('------------------------ RESULTS -------------------------');
strTemp = sprintf('NN      1-Tier      2-Tier     e-Measure     DCG\n');
disp(strTemp);
strTemp = sprintf('%.4f  %.4f      %.4f     %.4f        %.4f',avgFirst_NN,avgFirst_Tier,avgSecond_Tier,avgE_Measure,avgDCG);
disp(strTemp);
disp('----------------------------------------------------------');
disp(' ')

fclose(fp);
