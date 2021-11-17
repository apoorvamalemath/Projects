#distance="Canberra"

	
	for distance in {"L1","Euclidean","Cosine","EMD","Canberra"}	
	do	
		cd Matching		
		#python Distance_Shrec.py "$distance"
		cd ..
		pwd
		rm PerformanceEvaluation/"distance.txt"
		cp Matching/"distance.txt" PerformanceEvaluation/"distance.txt"
		cd PerformanceEvaluation/
		#rm -rf rankedLists
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


#,"Euclidean","Cosine","EMD","Canberra"
