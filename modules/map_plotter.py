import plotly.express as px
from typing import List
import pandas as pd
import numpy as np
import googlemaps
import os

"""
This module contains the MapPlotter class, that plots a map containing the most similar courses found with the SearchEngine class. The MapPlotter class has the following methods:

    - plot(resulting_dataset: pd.DataFrame) -> None: Function that plots the map with the most similar courses found.
"""
class MapPlotter():
    """
    Class that plots a map containing the most similar courses found with the SearchEngine class. The MapPlotter class has the following class variables:
    
            - GOOGLE_API_KEY: Google API key to use the Google Maps API (it is stored in a .env file in order to keep it private).
            - MAPBOX_API_KEY: Mapbox API key to use the Mapbox API (it is stored in a .env file in order to keep it private).

    """
    #Here we get the API keys from the .env file
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
    MAPBOX_API_KEY = os.environ['MAPBOX_API_KEY']

    def __init__(self, processed_dataset: pd.DataFrame):
        """
        Function that initializes the MapPlotter class.

        Args:
            - processed_dataset: Processed dataset containing all the information about the courses.
        """
        #Here we define the attributes of the class
        ##The dataset is the processed dataset
        self.processed_dataset = processed_dataset
        #We create a Google Maps client. This client will be used to get the geographical information of the courses from the Google Maps API
        self.gmaps_client = googlemaps.Client(key=self.GOOGLE_API_KEY)

    def plot(self, resulting_dataset: pd.DataFrame) -> None:
        """
        Function that plots the map with the most similar courses found.

        Args:
            - resulting_dataset: Dataset containing the most similar courses found.
        """
        #First we check if we've already obtained the geographical information of the courses.
        #This is to avoid making unnecessary calls to the Google Maps API since it has a limit of calls without paying.
        if not self.__check_lat_column(resulting_dataset):
            #If we haven't obtained the geographical information of the courses, we call the __process_result method to obtain it.
            self.__process_result(resulting_dataset)

        #We create the map using the plotly.express library. We use the scatter_mapbox method to create the map with the following information:
        #1) The latitude and longitude of the courses.
        #2) The name of the courses.
        #3) The similarity of the courses (the size of the points).
        #4) The fees of the courses (the color of the points).
        #5) The custom data of the courses (the textual university name, the faculty name, the fees and the duration).
        fig = px.scatter_mapbox(
        resulting_dataset,
        lat="lat",
        lon="lng",
        hover_name="courseName",
        size="similarity",
        color="fees (EUR)",
        range_color=[np.nanmin(self.processed_dataset["fees (EUR)"]), np.nanmax(self.processed_dataset["fees (EUR)"])],
        opacity=0.7,
        color_continuous_scale=px.colors.sequential.Turbo,
        custom_data=["universityName", "facultyName", "fees", "duration"],
        )
        #We update the layout of the map with the following information:
        #1) The title of the map.
        #2) The width and height of the map.
        #3) The background color of the map.
        #4) The style and zoom of the map. We use the Mapbox API to create the map since it allows us to create a map with a different style than the default one.
        #5) The margin of the map.
        #6) The style of the colorbar.
        fig.update_layout(
        title={
            'text': 'Most Similar Courses',
            'x': 0.5, 
            'font': {'size': 24, 'color': 'black' }
            },
        width=1200, 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600, 
        mapbox={
            "style": "light"
            }, 
        mapbox_accesstoken=self.MAPBOX_API_KEY, 
        margin={
            "t":50,
            "b":0,
            "l":0,
            "r":0
            }, 
        hovermode='closest', 
        coloraxis_colorbar={
            "title":"Fees (EUR)", 
            "x":-5, 
            "y":0.5, 
            "orientation":"v", 
            "title_font" : {"size":18, "color":"black"}, 
            "tickfont" : {"size":14, "color":"black"}
            }
        )
        #Finally, we update the hovertemplate of the map to show the custom data of the courses when we hover over them.
        fig.update_traces(
            hovertemplate="<br>".join(["%{hovertext}","University: %{customdata[0]}","Faculty: %{customdata[1]}","Fees: %{customdata[2]}","Duration: %{customdata[3]}"])
            )
        #We show the map.
        fig.show()


    def __process_result(self, resulting_dataset: pd.DataFrame) -> None:
        """
        Function that obtains the geographical information of the courses from the Google Maps API along with the custom data of the courses.

        Args:
            - resulting_dataset: Dataset containing the most similar courses found.
        """
        #We create a list with the indexes of courses in the resulting dataset.
        index_list = list(resulting_dataset.index)
        #Then, we obtain custom data of the courses from the processed dataset and we add it to the resulting dataset.
        #This custom data is: city, country, fees (text), duration and faculty name.
        self.__set_custom_info(resulting_dataset, index_list)
        #Once we have the custom data, we obtain the geographical information of the courses from the Google Maps API.
        #This geographical information is: full address, latitude and longitude.
        self.__set_geographical_info(resulting_dataset)

        #We return None since operations are done in place.
        return None

    def __check_lat_column(self, resulting_dataset: pd.DataFrame) -> bool:
        """
        Function that checks if the resulting dataset already contains the latitude column.

        Args:
            - resulting_dataset: Dataset containing the most similar courses found.

        Returns:
            - True if the resulting dataset already contains the latitude column.
            - False if the resulting dataset doesn't contain the latitude column.

        """
        #We check if the resulting dataset contains the latitude column.
        if "lat" in resulting_dataset.columns:
            #If it does, we return True.
            return True
        else:
            #If it doesn't, we return False.
            return False

    

    def __set_custom_info(self, resulting_dataset: pd.DataFrame, index_list: List[int]) -> None:
        """
        Function that obtains the custom data of the courses from the processed dataset and adds it to the resulting dataset.

        Args:
            - resulting_dataset: Dataset containing the most similar courses found.
            - index_list: List with the indexes of courses in the resulting dataset.
        """
        #Here we obtain the city field of the courses in the resulting dataset.
        resulting_dataset["city"] = self.__get_custom_info(index_list, "city")
        #Here we obtain the country field of the courses in the resulting dataset.
        resulting_dataset["country"] = self.__get_custom_info(index_list, "country")
        #Here we obtain the fees field of the courses in the resulting dataset.
        resulting_dataset["fees"] = self.__get_custom_info(index_list, "fees")
        #Here we obtain the numerical fees field of the courses in the resulting dataset.
        resulting_dataset["fees (EUR)"] = self.__get_custom_info(index_list, "fees (EUR)")
        #Here we obtain the duration field of the courses in the resulting dataset.
        resulting_dataset["duration"] = self.__get_custom_info(index_list, "duration")
        #Here we obtain the faculty name field of the courses in the resulting dataset.
        resulting_dataset["facultyName"] = self.__get_custom_info(index_list, "facultyName")

        #We return None since operations are done in place.
        return None

    def __get_custom_info(self, index_list: List[int], column_name: str) -> List[str]:
        """
        Function that obtains the custom data of the courses from the processed dataset.

        Args:
            - index_list: List with the indexes of courses in the resulting dataset.
            - column_name: Name of the column to obtain from the processed dataset.

        Returns:
            - List with the custom data of the courses.
        """
        #Here we obtain the custom data of the courses from the processed dataset.
        custom_info_list = list(self.processed_dataset.iloc[index_list][column_name])

        #We return the list with the custom data of the courses.
        return custom_info_list

    def __set_geographical_info(self, resulting_dataset: pd.DataFrame) -> None:
        """
        Function that obtains the geographical information of the courses from the Google Maps API.

        Args:
            - resulting_dataset: Dataset containing the most similar courses found.
        """
        #First we create a new column in the resulting dataset with the full address of the courses in order to obtain the geographical information from each address.
        resulting_dataset["fullAddress"] = resulting_dataset['facultyName'] + ',' + resulting_dataset['universityName'] + ',' + resulting_dataset['city'] + ',' + resulting_dataset['country']

        #Then, we obtain the geographical information of the courses from the Google Maps API. This information is the latitude and longitude of the address.
        #We use the gmaps_client created in the __init__ method to make the calls to the Google Maps API.
        resulting_dataset["lat"] = resulting_dataset["fullAddress"].apply(lambda x: self.gmaps_client.geocode(x)[0]["geometry"]["location"]["lat"])
        resulting_dataset["lng"] = resulting_dataset["fullAddress"].apply(lambda x: self.gmaps_client.geocode(x)[0]["geometry"]["location"]["lng"])

        #We return None since operations are done in place.
        return None
