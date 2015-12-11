#!/bin/bash


for ((x=1; x<=10; x++))
do
   for y in 0 0.5 1 1.5 2.0
   do
     for z in 3 7 9 15
      do
         printf "%d %f %d\n" $x $y $z
     done
   done
done

declare -a dataset=("USPS_1_rest" "USPS_1_1" "MNIST_1_rest" "MNIST_1_rest" )

for ds in "${dataset[@]}"
	for algorithm in "FO" "FO_ag" "FO_pr" "FO_ad" "SO"
		for b in 1.0e-07	1.0e-06	1.0e-05	0.0001	0.001	0.01	0.1	1	10	100	1000	10000	100000
		         printf "%s %s %f\n" $ds $algorithm $b
		done
	done
done

declare -a dataset=("USPS_1_rest" "USPS_1_1" "MNIST_1_rest" "MNIST_1_rest" )
for ds in "${dataset[@]}"
	printf %s\n $ds
done

declare -a arr=("USPS_1_rest" "USPS_1_1" "MNIST_1_rest" "MNIST_1_rest")

## now loop through the above array
for i in "${arr[@]}"
do
   printf "$s\n" $i
   # or do whatever with individual element of the array
done


