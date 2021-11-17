#!/bin/bash

echo "-----------------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
echo "Dataset: Shrec13, Features: FPFH, Distance: L1" >> PerformanceEvaluation/log_FV.txt
echo "-----------------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt

dimension=33
datasetPath="../features/Shrec13"
GMMVocabularyPath="../features/GMM_Shrec13"
#eachClassObjects=24
resultFileName="log_FV.txt"

cd FeatureEncoding/FV
python Octave_save.py $dimension $datasetPath
python Octave_save_vocabulary.py $dimension $GMMVocabularyPath
cd ../..

for k in 5
do 
	echo "K = $k"
	echo "-------------------------------------"
	cd FeatureEncoding/FV
	octave ./FV_K.m $k
	python Octave_load_Shrec.py
	cp "encoding.txt" ../../Matching/"encoding.txt"

	echo "Feature Enconding done successfully!!!"
	echo "-------------------------------------"
done
