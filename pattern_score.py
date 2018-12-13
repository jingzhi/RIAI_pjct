import csv
from sys import argv
# input explaination(e.g.: python3 pattern_score.py 3_10 '[True, True, True]')
# first arg is net name for e.g.: 3_10
# second arg is linear pattern for e.g.: '[True, True, True]'
i=0
j=0
k=0
with open('log.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
    	if line[0]==argv[1] and line[5]==argv[2]:
    		if line[3]=='True':	
	    		if line[4]=='True':
	    			i+=1
    			else:
    				j+=1
    				print("not verified img",line[2])
    		else:
    			k+=1
print("verified: {}, not verified: {}, not considered: {}".format(i,j,k))