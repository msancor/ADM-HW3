#!/bin/bash

#Script that merges all the .tsv files contained in the data/tsvs folder and prints the following information:
#1) The country that offers most courses
#2) The city that offers most courses
#3) The number of universities that offer part-time courses
#4) The percentage of courses in Engineering

#Here, we create a .tsv file concatenating all the .tsv files contained in the data/tsvs folder and saving it as merged_courses.tsv
#As a first step we add the header to the file
echo -e "courseName\tuniversityName\tfacultyName\tisItFullTime\tdescription\tstartDate\tfees\tmodality\tduration\tcity\tcountry\tadministration\turl" > data/merged_courses.tsv
#Then, we concatenate all the .tsv files contained in the data/tsvs folder
cat data/tsvs/course_{1..6000}.tsv >> data/merged_courses.tsv

#Here we print the country that offers most courses. We do this following these steps:
#1) We cut the 11th column of the merged_courses.tsv file, which contains the country name using the cut -f11 command
#2) We remove the header of the file using the tail -n +2 command
#3) We sort the file using the sort command since we need to count the number of times each country appears
#4) We count the number of times each unique country appears using the uniq -c command
#5) We sort the file in reverse order and by the number of times each country appears using the sort -rn command
#6) We print the first line of the file using the head -n 1 command
echo -n "The country that offers most courses is: " && cut -f11 data/merged_courses.tsv | tail -n +2 | sort | uniq -c | sort -rn | head -n 1 | awk '{$1=""; print $0}'
#Here we print the city that offers most courses. The steps are the same as the ones used for the country
echo -n "The city that offers most courses is: " && cut -f10 data/merged_courses.tsv | tail -n +2 | sort | uniq -c | sort -rn | head -n 1 | awk '{$1=""; print $0}'
#Here we print the number of universities that offer part-time courses. We do this following these steps:
#1) We cut the 2nd and 4th column of the merged_courses.tsv file, which contains the university name and the modality of the course using the cut -f2,4 command
#2) We remove the header of the file using the tail -n +2 command
#3) We select only the part-time courses using the grep "Part time" command
#4) We cut the university name using the cut -f1 command since we need to count the number of universities that offer part-time courses
#5) We sort the file using the sort command since we need to count the number of times each university appears
#6) We count the number of times each unique university appears using the uniq -c command
#7) We count the number of unique universities using the wc -l command
echo -n "The number of universities that offer part-time courses is: " && cut -f2,4 data/merged_courses.tsv | tail -n +2 | grep "Part time" | cut -f1 | sort | uniq | wc -l | awk '{print $1}'
#Here we print the percentage of courses in Engineering. We do this following these steps:
#1) We cut the 1st column of the merged_courses.tsv file, which contains the course name using the cut -f1 command
#2) We remove the header of the file using the tail -n +2 command
#3) We select only the courses that contain the word "engineer/Engineer" using a Regex and the grep "[eE]ngineer" command
#4) We count the number of courses that contain the word "engineer/Engineer" using the wc -l command
echo -n "The percentage of courses in Engineering is: " && cut -f1 data/merged_courses.tsv | tail -n +2 | grep "[eE]ngineer" | wc -l | awk '{print ($1/6000)*100 "%"}'
