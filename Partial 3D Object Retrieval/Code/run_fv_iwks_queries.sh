#!/bin/bash

echo "-----------------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
echo "Dataset: Shrec13, Features: FPFH, Distance: L1" >> PerformanceEvaluation/log_FV.txt
echo "-----------------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt

dimension=100
datasetPath="../features_iwks/Queries"
GMMVocabularyPath="../features_iwks/GMM_Vocabulary_Q"
#eachClassObjects=24
resultFileName="log_FV.txt"

cd FeatureEncoding/FV_Queries
python Octave_save.py $dimension $datasetPath
python Octave_save_vocabulary.py $dimension $GMMVocabularyPath
cd ../..

for k in 5
do 
	echo "K = $k" >> PerformanceEvaluation/log_FV.txt
	echo "--------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
	cd FeatureEncoding/FV_Queries
	octave ./FV_K.m $k
	python Octave_load_Shrec.py
	cp "encoding_Q.txt" ../../Matching/"encoding_Q.txt"

	echo "Feature Enconding done successfully!!!" >> PerformanceEvaluation/log_FV.txt
	echo "--------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
done
