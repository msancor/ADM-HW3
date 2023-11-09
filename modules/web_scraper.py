import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import os
import time

class WebScraper():
    def __init__(self, base_url: str = "https://www.findamasters.com", sub_url: str = "/masters-degrees/msc-degrees/?PG=", page_limit: int = 400):
        """
        Class that performs web scraping on multiple pages of a website.

        Attributes:
            base_url (str): Base url of the website to scrape.
            sub_url (str): Sub url of the website to scrape.
            page_limit (int): Number of pages to scrape.
        """
        #This is the base url of the website to scrape
        self.base_url = base_url
        #This is the sub url of the website to scrape
        self.sub_url = sub_url
        #This is the number of pages to scrape
        self.page_limit = page_limit

    def scrape_urls(self, html_class: str = "courseLink text-dark", save_path: str = "./data/"):
        """
        Function that scrapes the urls within an HTML class for multiple pages of a website and saves them in a text file.
        
        Args:
            html_class (str): HTML class of the urls to scrape.
            save_path (str): Path to save a text file containing the urls of all the courses.

        Returns:
            None
        """
        s = requests.Session()
        for page in range(1, self.page_limit + 1):
            html = s.get(self.base_url + self.sub_url + str(page), 
                         headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
                         ).text
            #This is the soup of the page to scrape
            soup = BeautifulSoup(html, 'html.parser')
            #This is the list of all the courses in the page to scrape
            courses = soup.find_all("a", {"class": html_class})
            #This is the list of all the urls of the courses in the page to scrape
            urls = [course.get("href") for course in courses]
            #This is the path to save the text file containing the urls of all the courses
            path = save_path + "urls.txt"
            #This creates a text file containing the urls of all the courses.
            with open(path, "a") as file:
                for url in urls:
                    file.write(self.base_url + url + "\n")   
            time.sleep(1)
        s.close()

    def save_htmls(self, save_path: str = "./data/"):
        """
        Function that saves the HTMLs of multiple pages of a website in a text file.
        
        Args:
            save_path (str): Path to save a text file containing the HTMLs of all the pages.

        Returns:
            None
        """
        urls = open(save_path + "urls.txt", "r").readlines()
        number_urls_per_folder = len(urls) // self.page_limit
        folder_path = save_path + "htmls/html_page1/"
        s = requests.Session()
        for i, url in enumerate(urls):
            try:
                html = s.get(url.strip(), 
                        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
                        )
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error: {e}. Waiting 60 seconds and trying again.")
                time.sleep(60)
                html = s.get(url.strip(), 
                        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}
                        )
            if i % number_urls_per_folder == 0:
                folder_path = save_path + "htmls/html_page" + str(i // number_urls_per_folder + 1) + "/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            
            with open(folder_path + "html" + str(i + 1) + ".html", "w") as file:
                file.write(html.text)
            file.close()
            time.sleep(5)
        s.close()
            
        