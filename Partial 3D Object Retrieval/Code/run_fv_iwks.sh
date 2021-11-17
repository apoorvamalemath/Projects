#!/bin/bash

echo "-----------------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
echo "Dataset: Shrec12, Features: IWKS" >> PerformanceEvaluation/log_FV.txt
echo "-----------------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt

dimension=100
datasetPath="../features_iwks/Shrec11"
GMMVocabularyPath="../features_iwks/GMM_Vocabulary_Shrec11"
resultFileName="log_FV.txt"

cd FeatureEncoding/FV
python Octave_save.py $dimension $datasetPath
python Octave_save_vocabulary.py $dimension $GMMVocabularyPath
cd ../..

datasetPath="../features_iwks/Queries"
GMMVocabularyPath="../features_iwks/GMM_Vocabulary_Q"
resultFileName="log_FV.txt"

cd FeatureEncoding/FV_Queries
python Octave_save.py $dimension $datasetPath
python Octave_save_vocabulary.py $dimension $GMMVocabularyPath
cd ../..

for k in 5 10 50 100 200 250 500
do 
	echo "K = $k" >> PerformanceEvaluation/log_FV.txt
	echo "-------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
	cd FeatureEncoding/FV
	octave ./FV_K.m $k
	python Octave_load_Shrec12.py
	cp "encoding.txt" ../../Matching/"encoding.txt"

	echo "Feature Enconding done successfully!!!" >> PerformanceEvaluation/log_FV.txt
	echo "---------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
	echo "Feature encoding for QUERIES" >> PerformanceEvaluation/log_FV.txt
	cd ../..		
	echo "K = $k" >> PerformanceEvaluation/log_FV.txt
	echo "--------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
	cd FeatureEncoding/FV_Queries
	octave ./FV_K.m $k
	python Octave_load_Shrec12.py
	cp "encoding_Q.txt" ../../Matching/"encoding_Q.txt"
	cd ..
	echo "Feature Enconding done successfully!!!" >> PerformanceEvaluation/log_FV.txt
	echo "--------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
	for distance in {"L1","Euclidean","Cosine","EMD"}	
	do	
		cd Matching		
		python Distance_Shrec12.py "$distance"
		cd ..
		pwd
		rm PerformanceEvaluation/"distance.txt"
		cp Matching/"distance.txt" PerformanceEvaluation/"distance.txt"
		cd PerformanceEvaluation/
		rm -rf rankedLists
		mkdir rankedLists/
		cp generateRankedList.py rankedLists/generateRankedList.py
		cd rankedLists
		pwd
		python generateRankedList.py
		rm generateRankedList.py
		cd ..	
		octave evaluate_range_rank_list.m 
		rm -rf rankedLists		
		cd ..
	done
	echo "Evaluation done successfully!!!"
	echo "----------------------------------------------------------------" >> PerformanceEvaluation/log_FV.txt
done
