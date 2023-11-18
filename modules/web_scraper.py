#Here we import the libraries we will use.
from bs4 import BeautifulSoup
import requests
import time
import os

"""
This module contains the WebScraper class that performs web scraping on multiple pages of a website. The WebScraper class contains two methods:

    - scrape_urls(): Scrapes the urls within an HTML class for multiple pages of a website and saves them in a text file.
    - scrape_htmls(): Scrapes and saves the HTMLs of multiple pages of a website in a text file.

"""

#Here we create the WebScraper class.
class WebScraper():
    """
    Class that performs web scraping on multiple pages of a website. The WebScraper class contains the following class variables:
    
        - BASE_URL (str): Base url of the website.
        - SUB_URL (str): Sub url of the website.
        - PAGE_LIMIT (int): Number of pages to scrape.
        - DATA_SAVE_PATH (str): Path to save the data.

    """
    #Here we define the class variables.
    #We use the website https://www.findamasters.com/ to scrape the urls and HTMLs of the courses.
    #The BASE_URL is the base url of the website.
    BASE_URL = "https://www.findamasters.com"
    #The SUB_URL is the sub url of the website. We use this url to scrape the urls and HTMLs of the courses.
    #This SUB_URL is the url for a given page of the website. For example, the url for the first page is https://www.findamasters.com/masters-degrees/msc-degrees/?PG=1.
    #The PG argument is the page number. We use this argument to loop through all the pages of the website.
    SUB_URL = "/masters-degrees/msc-degrees/?PG="
    #The PAGE_LIMIT is the number of pages we want to scrape. In this case, we are scraping the first 400 pages of the website.
    PAGE_LIMIT = 400
    #The DATA_SAVE_PATH is the path where we will save the data. In this case, we are saving the data in the data folder.
    DATA_SAVE_PATH = "./data/"

    def __init__(self):
        """ 
        Function that initializes the class.
        """
        #Here we initialize the class. Since it is empty, we use the pass statement.
        pass

    def scrape_urls(self, html_class: str = "courseLink"):
        """
        Function that scrapes the urls within an HTML class for multiple pages of a website and saves them in a text file.
        
        Args:
            html_class (str): HTML class of the urls to scrape. Its default value is "courseLink" since we wanted to extract the urls of all the courses in the first 400 pages of the website.

        Returns:
            None
        """
        #First we create a session in order to save the cookies and not get blocked by the website. 
        s = requests.Session()

        #Here we loop through all the pages we want to scrape. In this case, we are scraping the first 400 pages of the website.
        for page in range(1, self.PAGE_LIMIT + 1):
            
            #Here we obtain the HTML text of the ith page we want to scrape. We use the headers argument to avoid getting blocked by the website.
            #The User-Agent header is used to identify the client making the request to the server. In this case, we are using the User-Agent header of a browser (mine).
            #We did this since we kept obtaining a 403 error: Forbidden. This error is caused by the website blocking our request.
            html = s.get(self.BASE_URL + self.SUB_URL + str(page),
                         headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
                        ).text
            #Here we create a BeautifulSoup object from the HTML text of the ith page we want to scrape and parse it using the html.parser.
            soup = BeautifulSoup(html, 'html.parser')
            #In order to find the urls of all the courses in the page, we find all the "a" tags with the class "courseLink". We know this by inspecting the HTML of the website.
            courses = soup.find_all("a", {"class": html_class})
            #After finding all the "a" tags with the class "courseLink", we extract the href attribute of each tag. This attribute contains the url of the course.
            urls = [course.get("href") for course in courses]
            
            #Here we save the urls in a text file on the path .data/urls.txt. We use the "a" mode to append the urls of the ith page to the text file since we are looping through all the pages.
            with open(self.DATA_SAVE_PATH + "urls.txt", "a") as file:
                #Here we loop through all the urls of the ith page and write them in the text file. We add the base url to the url since the urls in the website are relative.
                for url in urls:
                    file.write(self.BASE_URL + url + "\n")   

            #After scraping the urls of the ith page, we wait 1 second before scraping the urls of the next page. This is done to avoid getting blocked by the website.
            #We did this since we kept obtaining Error 429: Too Many Requests. This error is caused by the website blocking our request.
            time.sleep(1)
        
        #After scraping the urls of all the pages, we close the session.
        s.close()

    def scrape_htmls(self):
        """
        Function that scrapes and saves the HTMLs of multiple pages of a website in a text file.
        
        Args:
            None

        Returns:
            None
        """
        #First we check if the urls.txt file exists. If it doesn't, we raise an error.
        assert os.path.exists(self.DATA_SAVE_PATH + "urls.txt"), "You need to scrape the urls first. Use the scrape_urls() method."

        #Here we read the urls.txt file and save the urls in a list.
        urls = open(self.DATA_SAVE_PATH + "urls.txt", "r").readlines()
        #Then, we extract how many URLs existed per page since we will save the HTMLs of each page in a different folder.
        #In the end we will have 400 folders, each containing the HTMLs of the courses in the page.
        number_urls_per_folder = len(urls) // self.PAGE_LIMIT
        #Now we initialize the folder path where we will save the HTMLs of the courses in the first page.
        folder_path = self.DATA_SAVE_PATH + "htmls/html_page1/"

        #Here we create a session in order to save the cookies and not get blocked by the website.
        s = requests.Session()

        #Here we loop through all the urls we want to scrape. In this case, we are scraping 6000 urls.
        for i, url in enumerate(urls):

            #Since we didn't had a stable internet connection, we added a try-except block to avoid getting ConnectionError.
            #First we try to get the HTML of the ith url. The User-Agent header is used to identify the client making the request to the server. 
            #In this case, we are using the User-Agent header of a Chrome browser (mine).
            #We did this since we kept obtaining a 403 error: Forbidden. This error is caused by the website blocking our request.
            try:
                html = s.get(url.strip(), 
                        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
                        )
            #If we get a ConnectionError, we wait 60 seconds and try again. This is done since our internet connection was unstable but it tended to work after a minute.
            except requests.exceptions.ConnectionError as e:
                #Here we print the error message.
                print(f"Connection error: {e}. Waiting 60 seconds and trying again.")
                #Here we wait 60 seconds.
                time.sleep(60)
                #Here we try to get the HTML of the ith url again.
                html = s.get(url.strip(), 
                        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
                        )
            #Now, if we already filled the folder with the HTMLs of the courses in the page, we create a new folder to save the HTMLs of the courses in the next page.
            #We do this by checking if the remainder of the division of the number of urls we scraped by the number of urls per page is 0.
            if i % number_urls_per_folder == 0:
                #Here we create a new folder path to save the HTMLs of the courses in the next page.
                folder_path = self.DATA_SAVE_PATH + "htmls/html_page" + str(i // number_urls_per_folder + 1) + "/"
                #If this path doesn't exist, we create it.
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            
            #Finally, within each folder we save the HTML of the ith url in a text file.
            with open(folder_path + "html" + str(i + 1) + ".html", "w") as file:
                file.write(html.text)
            file.close()

            #After scraping the HTML of the ith url, we wait 5 seconds before scraping the HTML of the next url. 
            #This is done since we kept getting HTMLS with content: "Just a moment...". This is caused by the website blocking our request since we were scraping too fast.
            time.sleep(5)

        #After scraping the HTMLs of all the pages, we close the session.
        s.close()    
