
	for file in {10..99}
	do
		python kmeans_3.py "features_iwks/Shrec11/D000$file.txt"	
		mv outfile "target/D000$file.txt"
	done
	
	for file in {1..9}
	do
		python kmeans_3.py "features_iwks/Shrec11/D0000$file.txt"	
		mv outfile "target/D0000$file.txt"
	done

	for file in {100..999}
	do
		python kmeans_3.py "features_iwks/Shrec11/D00$file.txt"	
		mv outfile "target/D00$file.txt"
	done

	python kmeans_3.py "features_iwks/Shrec11/D01000.txt"	
	mv outfile "target/D01000.txt"

	for file in {1..9}
	do
		python kmeans_3.py "features_iwks/Queries/RQ000$file.txt"	
		mv outfile "query/RQ000$file.txt"
	done

	for file in {10..99}
	do
		python kmeans_3.py "features_iwks/Queries/RQ00$file.txt"	
		mv outfile "query/RQ00$file.txt"
	done

	for file in {100..150}
	do
		python kmeans_3.py "features_iwks/Queries/RQ0$file.txt"	
		mv outfile "query/RQ0$file.txt"
	done
	
	python sqfd_1.py
