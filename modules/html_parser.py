#Here we import the libraries we use.
from bs4 import BeautifulSoup
import pandas as pd
import os

"""
This module contains the HTMLParser class. The HTMLParser class parses information within the HTMLs of multiple pages of the https://www.findamasters.com website. The HTMLParser class contains two methods:

    - parse_htmls(): Parses information within the HTMLs of multiple pages of the https://www.findamasters.com
    - get_dataframe(): Obtains a dataframe containing the parsed information of all the pages of the https://www.findamasters.com website.

""" 

class HTMLParser():
    """
    Class that parses information within the HTMLs of multiple pages of the https://www.findamasters.com website. The HTMLParser class contains the following class variables:

        - HTML_PATH (str): Path to a folder containing the HTMLs of all the pages.
        - TSV_PATH (str): Path to save a TSV file containing the parsed information of all the pages.
        - NUMBER_OF_PAGES (int): Number of pages to parse.
        - COURSES_PER_PAGE (int): Number of courses per page.
        - NUMBER_OF_COURSES (int): Number of courses to parse.
    """
    #Here we define the class variables.
    #The HTML_PATH is the path to a folder containing the HTMLs of all the pages.
    HTML_PATH = "./data/htmls/"
    #The TSV_PATH is the path to save a TSV file containing the parsed information of all the pages.
    TSV_PATH = "./data/tsvs/"
    #The NUMBER_OF_PAGES is the number of pages we want to parse. In this case, we are parsing the first 400 pages of the website.
    NUMBER_OF_PAGES = 400
    #The COURSES_PER_PAGE is the number of courses per page. In this case, there are 15 courses per page.
    COURSES_PER_PAGE = 15
    #The NUMBER_OF_COURSES is the number of courses to parse. In this case, we are parsing the first 6000 courses of the website.
    NUMBER_OF_COURSES = 6000

    def __init__(self):
        """
        Function that initializes the class.

        """
        #Here we initialize the class. Since it is empty, we use the pass statement.
        pass

    def parse_htmls(self):
        """
        Function that parses information within the HTMLs of multiple pages of the https://www.findamasters.com website and saves it in a TSV file.

        Args:
            None

        Returns:
            None
        """
        #First, we loop through all the pages we want to parse. In this case, we are parsing the first 400 pages of the website.
        for i in range(1,self.NUMBER_OF_PAGES+1):
            #Then, we loop through all the courses per page. In this case, there are 15 courses per page.
            for j in range(1,self.COURSES_PER_PAGE+1):
                #Here we open the HTML file of the ith page and jth course. We use the courses_per_page*(i-1)+ j to obtain the correct number of the course.
                with open(self.HTML_PATH + f"html_page{i}/" + f"html{self.COURSES_PER_PAGE*(i-1)+ j}.html", "r") as file:
                    #Here we parse the HTML file using BeautifulSoup and its default parser.
                    parsed_html_file = BeautifulSoup(file.read(), "html.parser")
                file.close()

                #Now we obtain the properties of every course on the ith page and jth course of the website.
                #First, we obtain the course name using a private function called __get_course_name(). A private function is a function that can only be called within the class.
                course_name = self.__get_course_name(parsed_html_file)
                #Then, we obtain the university name using a private function called __get_university_name().
                university_name = self.__get_university_name(parsed_html_file)
                #Then, we obtain the faculty name using a private function called __get_faculty_name().
                faculty_name = self.__get_faculty_name(parsed_html_file)
                #Then, we obtain if the course is full time using a private function called __is_full_time().
                is_full_time = self.__is_full_time(parsed_html_file)
                #Then, we obtain the description of the course using a private function called __get_description().
                description = self.__get_description(parsed_html_file)
                #Then, we obtain the start date of the course using a private function called __get_start_date().
                start_date = self.__get_start_date(parsed_html_file)
                #Then, we obtain the fees of the course using a private function called __get_fees().
                fees = self.__get_fees(parsed_html_file)
                #Then, we obtain the modality of the course using a private function called __get_modality().
                modality = self.__get_modality(parsed_html_file)
                #Then, we obtain the duration of the course using a private function called __get_duration().
                duration = self.__get_duration(parsed_html_file)
                #Then, we obtain the city of the course using a private function called __get_city().
                city = self.__get_city(parsed_html_file)
                #Then, we obtain the country of the course using a private function called __get_country().
                country = self.__get_country(parsed_html_file)
                #Then, we obtain the administration of the course using a private function called __get_administration().
                administration = self.__get_administration(parsed_html_file)
                #Then, we obtain the link of the course using a private function called __get_link().
                link = self.__get_link(parsed_html_file)

                #Finally, we save all the information for each course in a TSV file. We use the courses_per_page*(i-1)+ j to obtain the correct number of the course.
                with open(self.TSV_PATH+"course_"+str(self.COURSES_PER_PAGE*(i-1)+ j)+".tsv", "w") as tsv_file:
                    tsv_file.write(f"{course_name}\t{university_name}\t{faculty_name}\t{is_full_time}\t{description}\t{start_date}\t{fees}\t{modality}\t{duration}\t{city}\t{country}\t{administration}\t{link}\n")
                tsv_file.close()

    def get_dataframe(self) -> pd.DataFrame:
        """
        Function that obtains a dataframe containing the parsed information of all the pages of the https://www.findamasters.com website.

        Args:
            None

        Returns:
            dataframe (pd.DataFrame): Dataframe containing the parsed information of all the pages of the https://www.findamasters.com website.
        """

        #First, we check if the TSV_PATH exists. If it does not exist, we raise an error.
        assert os.path.exists(self.TSV_PATH), "The TSV_PATH does not exist. Please run the parse_htmls() method first."

        #Now, we create an empty dataframe.
        dataframe = pd.DataFrame()
        
        #For every course, we read the TSV file containing its information and concatenate it to the dataframe.
        for i in range(1, self.NUMBER_OF_COURSES+1):
            dataframe = pd.concat([dataframe, pd.read_csv(self.TSV_PATH + f"course_{i}.tsv", sep="\t", header=None)])

        #Here we name the columns of the dataframe.
        dataframe.columns = ["courseName", "universityName", "facultyName", "isItFullTime", "description", "startDate", "fees", "modality", "duration", "city", "country", "administration", "url"]
        #Here we reset the index of the dataframe.
        dataframe.reset_index(drop=True, inplace=True)

        #Finally, we return the dataframe.
        return dataframe

    def __get_course_name(self, parsed_html_file: BeautifulSoup, class_name: str = "course-header__course-title") -> str:
        """
        Private function that obtains the name of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the course name. Its default value is "course-header__course-title" since it is the class name of the course name for the https://www.findamasters.com website.

        Returns:
            course_name (str): Name of the course.

        """
        #To obtain the course name we use a try-except block. If the course name is not found, we return an empty string.
        try:
            #Here we obtain the course name.
            #We use the find() method to find the first h1 tag with the class name "course-header__course-title".
            #Once we find the h1 tag, we use the get() method to obtain the value of the data-permutive-title attribute, i.e. the course name.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            course_name = parsed_html_file.find("h1", {"class": class_name}).get("data-permutive-title").strip()
        except:
            #If the course name is not found, we return an empty string.
            course_name = ""
        
        #Finally, we return the course name.
        return course_name
    
    def __get_university_name(self, parsed_html_file: BeautifulSoup, class_name: str = "course-header__institution") -> str:
        """
        Private function that obtains the name of a university from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the university name. Its default value is "course-header__institution" since it is the class name of the university name for the https://www.findamasters.com website.

        Returns:
            university_name (str): Name of the university.

        """
        #To obtain the university name we use a try-except block. If the university name is not found, we return an empty string.
        try:
            #Here we obtain the university name.
            #We use the find() method to find the first a tag with the class name "course-header__institution".
            #Once we find the a tag, we use the text attribute to obtain the text of the tag, i.e. the university name.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            university_name = parsed_html_file.find("a", {"class": class_name}).text.strip()
        except:
            #If the university name is not found, we return an empty string.
            university_name = ""
        
        #Finally, we return the university name.
        return university_name
    
    def __get_faculty_name(self, parsed_html_file: BeautifulSoup, class_name: str = "course-header__department") -> str:
        """
        Private function that obtains the name of a faculty from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the faculty name. Its default value is "course-header__department" since it is the class name of the faculty name for the https://www.findamasters.com website.

        Returns:
            faculty_name (str): Name of the faculty.

        """
        #To obtain the faculty name we use a try-except block. If the faculty name is not found, we return an empty string.
        try:
            #Here we obtain the faculty name.
            #We use the find() method to find the first a tag with the class name "course-header__department".
            #Once we find the a tag, we use the text attribute to obtain the text of the tag, i.e. the faculty name.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            faculty_name = parsed_html_file.find("a", {"class": class_name}).text.strip()
        except:
            #If the faculty name is not found, we return an empty string.
            faculty_name = ""
        
        #Finally, we return the faculty name.
        return faculty_name
    
    def __is_full_time(self, parsed_html_file: BeautifulSoup, class_name: str = "key-info__study-type") -> str:
        """
        Private function that obtains if a course is full time from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the full time. Its default value is "key-info__study-type" since it is the class name of the full time for the https://www.findamasters.com website.

        Returns:
            is_full_time (str): If the course is full time.

        """
        #To obtain if the course is full time we use a try-except block. If the full time is not found, we return an empty string.
        try:
            #Here we obtain if the course is full time.
            #We use the find() method to find the first span tag with the class name "key-info__study-type".
            #Once we find the span tag, we use the a tag to obtain the text of the tag, i.e. if the course is full time.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            full_time = parsed_html_file.find("span", {"class": class_name}).find("a").text.strip()
        except:
            #If the full time is not found, we return an empty string.
            full_time = ""
        
        #Finally, we return if the course is full time.
        return full_time

    def __get_start_date(self, parsed_html_file: BeautifulSoup, class_name: str = "key-info__start-date") -> str:
        """
        Private function that obtains the start date of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the start date. Its default value is "key-info__start-date" since it is the class name of the start date for the https://www.findamasters.com website.

        Returns:
            start_date (str): Start date of the course.

        """
        #To obtain the start date we use a try-except block. If the start date is not found, we return an empty string.
        try:
            #Here we obtain the start date.
            #We use the find() method to find the first span tag with the class name "key-info__start-date".
            #Once we find the span tag, we use the text attribute to obtain the text of the tag, i.e. the start date.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            start_date = parsed_html_file.find("span", {"class": class_name}).text.strip()
        except:
            #If the start date is not found, we return an empty string.
            start_date = ""
        
        #Finally, we return the start date.
        return start_date
    
    def __get_modality(self, parsed_html_file: BeautifulSoup, class_name: str = "key-info__qualification") -> str:
        """
        Private function that obtains the modality of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the modality. Its default value is "key-info__qualification" since it is the class name of the modality for the https://www.findamasters.com website.

        Returns:
            modality (str): Modality of the course.

        """
        #To obtain the modality we use a try-except block. If the modality is not found, we return an empty string.
        try:
            #Here we obtain the modality.
            #We use the find() method to find the first span tag with the class name "key-info__qualification".
            #Once we find the span tag, we use the text attribute to obtain the text of the tag, i.e. the modality.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            full_time = parsed_html_file.find("span", {"class": class_name}).find("a").text.strip()
        except:
            #If the modality is not found, we return an empty string.
            full_time = ""
        
        #Finally, we return the modality.
        return full_time
    
    def __get_duration(self, parsed_html_file: BeautifulSoup, class_name: str = "key-info__duration") -> str:
        """
        Private function that obtains the duration of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the duration. Its default value is "key-info__duration" since it is the class name of the duration for the https://www.findamasters.com website.

        Returns:
            duration (str): Duration of the course.

        """
        #To obtain the duration we use a try-except block. If the duration is not found, we return an empty string.    
        try:
            #Here we obtain the duration.
            #We use the find() method to find the first span tag with the class name "key-info__duration".
            #Once we find the span tag, we use the text attribute to obtain the text of the tag, i.e. the duration.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            duration = parsed_html_file.find("span", {"class": class_name}).text.strip()
        except:
            #If the duration is not found, we return an empty string.
            duration = ""

        #Finally, we return the duration.    
        return duration
    
    def __get_city(self, parsed_html_file: BeautifulSoup, class_name: str = "course-data__city") -> str:
        """
        Private function that obtains the city of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the city. Its default value is "course-data__city" since it is the class name of the city for the https://www.findamasters.com website.

        Returns:
            city (str): City of the course.

        """
        #To obtain the city we use a try-except block. If the city is not found, we return an empty string.    
        try:
            #Here we obtain the city.
            #We use the find() method to find the first a tag with the class name "course-data__city".
            #Once we find the a tag, we use the text attribute to obtain the text of the tag, i.e. the city.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            city = parsed_html_file.find("a", {"class": class_name}).text.strip()
        except:
            #If the city is not found, we return an empty string.
            city = ""

        #Finally, we return the city.    
        return city
    
    def __get_country(self, parsed_html_file: BeautifulSoup, class_name: str = "course-data__country") -> str:
        """
        Private function that obtains the country of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the country. Its default value is "course-data__country" since it is the class name of the country for the https://www.findamasters.com website.

        Returns:
            country (str): Country of the course.

        """
        #To obtain the country we use a try-except block. If the country is not found, we return an empty string.
        try:
            #Here we obtain the country.
            #We use the find() method to find the first a tag with the class name "course-data__country".
            #Once we find the a tag, we use the text attribute to obtain the text of the tag, i.e. the country.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            country = parsed_html_file.find("a", {"class": class_name}).text.strip()
        except:
            #If the country is not found, we return an empty string.
            country = ""
        #Finally, we return the country.    
        return country
    
    def __get_administration(self, parsed_html_file: BeautifulSoup, class_name: str = "course-data__on-campus") -> str:
        """
        Private function that obtains the administration of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the administration. Its default value is "course-data__on-campus" since it is the class name of the administration for the https://www.findamasters.com website.

        Returns:
            administration (str): Administration of the course.

        """
        #To obtain the administration we use a try-except block. If the administration is not found, we return an empty string.        
        try:
            #Here we obtain the administration.
            #We use the find() method to find the first a tag with the class name "course-data__on-campus".
            #Once we find the a tag, we use the text attribute to obtain the text of the tag, i.e. the administration.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            administration = parsed_html_file.find("a", {"class": class_name}).text.strip()
        except:
            #If the administration is not found, we return an empty string.
            administration = ""

        #Finally, we return the administration.    
        return administration
    
    def __get_fees(self, parsed_html_file: BeautifulSoup, class_name: str = "course-sections__fees") -> str:
        """
        Private function that obtains the fees of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the fees. Its default value is "course-sections__fees" since it is the class name of the fees for the https://www.findamasters.com website.

        Returns:
            fees (str): Fees of the course.

        """
        #To obtain the fees we use a try-except block. If the fees is not found, we return an empty string.        
        try:
            #Here we obtain the fees.
            #We use the find() method to find the first div tag with the class name "course-sections__fees".
            #Once we find the div tag, we use the p tag to obtain the text of the tag, i.e. the fees.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            fees = parsed_html_file.find("div", {"class": class_name}).find("p").text.strip()
        except:
            #If the fees are not found, we return an empty string.
            fees = ""

        #Finally, we return the fees.        
        return fees
    
    def __get_description(self, parsed_html_file: BeautifulSoup, class_name: str = "course-sections__description") -> str:
        """
        Private function that obtains the description of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the description. Its default value is "course-sections__description" since it is the class name of the description for the https://www.findamasters.com website.

        Returns:
            description (str): Description of the course.

        """
        #To obtain the description we use a try-except block. If the description is not found, we return an empty string.
        try:
            #Here we obtain the description.
            #We use the find() method to find the first div tag with the class name "course-sections__description".
            #Once we find the div tag, we use the p tag to obtain the text of the tag, i.e. the description.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            description = parsed_html_file.find("div", {"class": class_name}).find_all("p", {"class": ""})
            #In this case we have to parse the description since we obtain a list of p tags with the .find_all() method.
            #We use the .join() method to join all the p tags in the list with a whitespace.
            #Importantly we use the .strip('"') method to remove the leading and trailing double quotes from each text line.
            parsed_description = " ".join([line.text.strip('"').strip() for line in description])
        except:
            #If the description is not found, we return an empty string.
            parsed_description = ""

        #Finally, we return the description.        
        return parsed_description
    
    def __get_link(self, parsed_html_file: BeautifulSoup, class_name: str = "canonical") -> str:
        """
        Private function that obtains the link of a course from its HTML.

        Args:
            parsed_html_file (BeautifulSoup): Parsed HTML of a course.
            class_name (str): Class name of the link. Its default value is "canonical" since it is the class name of the link for the https://www.findamasters.com website.

        Returns:
            link (str): Link of the course.

        """
        #To obtain the link we use a try-except block. If the link is not found, we return an empty string.
        try:
            #Here we obtain the link.
            #We use the find() method to find the first link tag with the class name "canonical".
            #Once we find the link tag, we use the get() method to obtain the value of the href attribute, i.e. the link.
            #Finally, we use the strip() method to remove the leading and trailing whitespaces.
            link = parsed_html_file.find("link", {"rel": class_name}).get("href").strip()
        except:
            #If the link is not found, we return an empty string.
            link = ""

        #Finally, we return the link.        
        return link
