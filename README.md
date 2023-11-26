# Algorithmic Methods for Data Mining - Homework 3

This is a Github repository created to submit the third Homework of the **Algorithmic Methods for Data Mining (ADM)** course for the MSc. in Data Science at the Sapienza University of Rome.

--- 
## What's inside this repository?

1. `README.md`: A markdown file that explains the content of the repository.

2. `main.ipynb`: A Jupyter Notebook file containing all the relevant exercises and reports belonging to the homework questions, the *Command Line Question*, and the *Algorithmic Question*.

3. ``modules/``: A folder including 4 Python modules used to solve the exercises in `main.ipynb`. The files included are:

    - `__init__.py`: A *init* file that allows us to import the modules into our Jupyter Notebook.

    - `web_scraper.py`: A Python file including a `WebScraper` class designed to perform web scraping on the multiple pages of the [MSc. Degrees](https://www.findamasters.com/masters-degrees/msc-degrees/) website.

    - `html_parser.py`: A Python file including a `HTMLParser` class designed to parse the HTML files obtained by the web scraping process and extract relevant information.

    - `data_preprocesser.py`: A Python file including a `DataPreprocesser` class designed to pre-process text data in order to obtain information and build a Search Engine.

    - `search_engine.py`: A Python file including three classes: `SearchEngine`, `TopKSearchEngine`, and `WeightedTopKSearchEngine` designed to implement different versions of a Search Engine that queries information from our MSc. courses dataset.

    - `map_plotter.py`: A Python file including a `MapPlotter` class designed to plot a map including the results of our Search Engines.

4. `CommandLine.sh`: A bash script including the code to solve the *Command Line Question*.

5. ``.gitignore``: A predetermined `.gitignore` file that tells Git which files or folders to ignore in a Python project.

6. `LICENSE`: A file containing an MIT permissive license.

## Datasets

The data used to work in this repository was obtained by performing web-scraping in the [MSc. Degrees](https://www.findamasters.com/masters-degrees/msc-degrees/) website via the `WebScraper` class contained in the `web_scraper.py` module. If you want to reproduce the data perform the following steps:

**1.** Create the directories where you will save the obtained `html` and `tsv` files after you perform web-scraping. Specifically, run in your terminal the following commands:

```bash
mkdir data
mkdir data/htmls
mkdir data/tsvs
```

Make sure you create these folders in the same directory you've saved the `main.ipynb` file on.

**2.** Open the `main.ipynb` file and run the cells contained in the **Data Collection** section and you will obtain all the pertinent data files.

Alternatively, you can also just create a new `.py` file and run the following script:

```python
from modules.web_scraper import WebScraper
from modules.html_parser import HTMLParser

#First, we have to initialize the WebScraper and HTMLParser classes by calling their constructors
web_scraper = WebScraper()
html_parser = HTMLParser()

#Here, the method saves scraped URLs from the website in a text file called urls.txt
web_scraper.scrape_urls()

#Here, we can call the .scrape_htmls() method to get the HTMLs of the URLs mentioned above.
web_scraper.scrape_htmls()

#Finally, you can parse the obtained HTMLs to obtain information about the courses and save it in .tsv files.
html_parser.parse_htmls()

#If you want to visualize the information obtained, you can store it in a dataframe
df = html_parser.get_dataframe()
df.head()
```

Take into account this script will take $\sim 11$ hours so run it only if necessary.

## API Usage

In order to plot a map with our results using the `MapPlotter` class defined before we used two external API's:

- [Google Maps API](https://mapsplatform.google.com/pricing/)

- [Mapbox API](https://docs.mapbox.com/api/overview/)

In order to gain access to these API's we had to create accounts and obtain **API keys** from each API. These keys were stored in a `.env` file and then read directly into the `map_plotter.py` module. Because of privacy issues this file is not included in our repository but can be recreated by any user with their own API keys. To do this follow these steps:

1. Obtain API keys for both the [Google Maps API](https://mapsplatform.google.com/pricing/) and [Mapbox API](https://docs.mapbox.com/api/overview/).

2. Create a `.env` file including the following:

    ```shell
    GOOGLE_API_KEY =
    MAPBOX_API_KEY =
    ```

    and write your API keys in the other side of the equal sign.


## Important Note

If the Notebook doesn't load through Github please try all of these steps:

1. Try compiling the Notebook through its NBViewer.

2. Try downloading the Notebook and opening it in your local computer.

3. Try opening it through Google Colab through the following link.

**NOTE:** Since Plotly Express maps are not compiled well through NBViewer or Github, we recommend downloading the Notebook for a better visualization.

---

**Author:** Miguel Angel Sanchez Cortes

**Email:** sanchezcortes.2049495@studenti.uniroma1.it

*MSc. in Data Science, Sapienza University of Rome*
