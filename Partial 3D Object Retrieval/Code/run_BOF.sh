#!/bin/bash

echo "-----------------------------------------------------------------------" >> PerformanceEvaluation/log_BoF.txt
echo "Dataset: Shrec 11, Features: IWKS" >> PerformanceEvaluation/log_BoF.txt
echo "-----------------------------------------------------------------------" >> PerformanceEvaluation/log_BoF.txt

dimension=100
datasetPath="../features_iwks/Shrec11"
#eachClassObjects=24
resultFileName="log_BoF.txt"

datasetPath2="../features_iwks/Queries"

for k in 5 10 25 50 75 100 150 200 300 400 500 1000 
do 
	echo "K = $k"
	echo "-------------------------------------"
	cd FeatureEncoding/BoF
	python BoF_Shrec12.py $dimension $datasetPath $k
	cp "encoding.txt" ../../Matching/"encoding.txt"
	cd ../..
	cd FeatureEncoding/BoFQ
	python BoF_Shrec12.py $dimension $datasetPath2 $k
	cp "encoding_Q.txt" ../../Matching/"encoding_Q.txt"
	echo "Feature Enconding done successfully!!!"
	echo "-------------------------------------"
	echo "-----------------------------------------------------" >> log_BoF.txt
	cd ../..
	bash distance.sh
	echo "-------------------------------------"
done
