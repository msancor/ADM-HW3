from modules.data_preprocesser import DataPreprocesser
from typing import List, Tuple, Dict
from collections import defaultdict
import pandas as pd
import pickle
import os

class SearchEngine():
    COLUMN_TO_QUERY = "description"

    def __init__(self, processed_dataset: pd.DataFrame):
        """
        Function that initializes the class.

        Args:
            processed_dataset (pd.DataFrame): Processed dataset i.e. the dataset after applying the DataPreprocesser class.
        """
        #Here we define the attributes of the class
        #The dataset is the processed dataset
        self.dataset = processed_dataset
        #The vocabulary is the vocabulary of the dataset i.e. a dictionary with unique words as keys and the index as values
        self.vocabulary = self.__get_vocabulary()
        #The inverted index is the inverted index of the dataset i.e. a dictionary with the index of the word as keys and the index of the documents as values
        self.inverted_index = self.__get_inverted_index()

    def query(self, query_text: str) -> pd.DataFrame:
        """
        Function that queries the dataset.

        Args:
            query (str): Query to be searched in the dataset.

        Returns:
            results (pd.DataFrame): Results of the query.
        """
        #First we preprocess the query
        query_text = DataPreprocesser(self.dataset).get_processed_text(query_text)

        #Now we split the query into word tokens and obtain only the unique ones
        query_text = list(set(query_text.split(",")))

        #Now we get the index of the documents that contain ALL the words in the query since it is a conjunctive query
        documents_index = self.__get_document_indexes(query_text)

        #Finally we filter the dataset to get the results of the query
        results = self.dataset.iloc[documents_index][["courseName", "universityName", "description", "url"]]

        #Finally we return the results sorted by index
        return results.sort_index()
    
    def __get_document_indexes(self, query_tokens: List[str]) -> List[int]:
        """
        Function that gets the index of the documents that contain all the words in a text query.

        Args:
            query_tokens (List[str]): Query tokens.

        Returns:
            documents_index (List[int]): Index of the documents that contain ALL the words in the query.
        """
        #First we initialize a list that will contain the index of the documents that contain ALL the words in the query
        documents_index = []
        #We iterate over the words in the query
        for word in query_tokens:
            #If the word is not in the vocabulary, we return an empty list since all the words in the query must be in the vocabulary
            if word not in self.vocabulary.keys():
                return []
            #If the word is in the vocabulary, we add the index of the documents that contain it to the list
            else:
                documents_index.append(self.inverted_index[self.vocabulary[word]])

        #Now we get the intersection of the lists in the documents_index list
        documents_index = list(set(documents_index[0]).intersection(*documents_index))

        #Finally we return the index of the documents that contain ALL the words in the query
        return documents_index

    def __get_vocabulary(self) -> Dict[str, int]:
        """
        Function that gets the vocabulary of the dataset.
        
        Args:
            None

        Returns:
            vocabulary (Dict[str, int]): Vocabulary of the dataset.
        """
        #First we check if the vocabulary has been created before
        if os.path.isfile(f"data/{self.COLUMN_TO_QUERY}_vocabulary.pickle"):
            #If it has been created before, we load it
            with open(f"data/{self.COLUMN_TO_QUERY}_vocabulary.pickle", "rb") as f:
                vocabulary = pickle.load(f)
        #If it has not been created before, we create it
        else:
            #We create the vocabulary by calling the __create_vocabulary function
            vocabulary = self.__create_vocabulary()

        #We return the vocabulary
        return vocabulary
    
    def __create_vocabulary(self) -> Dict[str, int]:
        """
        Function that creates the vocabulary of the dataset.

        Args:
            None

        Returns:
            vocabulary (Dict[str, int]): Vocabulary of the dataset.
        """
        #First we get the unique words in the dataset sorted alphabetically
        unique_words = self.__get_unique_words()

        #After this, we initialize a dictionary that will contain the vocabulary
        vocabulary = {}
        #We iterate over the unique words and add them to the vocabulary
        for i, word in enumerate(unique_words):
            vocabulary[word] = i

        #Finally we save it to a file
        with open(f"data/{self.COLUMN_TO_QUERY}_vocabulary.pickle", "wb") as f:
            pickle.dump(vocabulary, f)

        #We return the vocabulary
        return vocabulary
    
    def __get_unique_words(self) -> List[str]:
        """
        Function that gets the unique words in the dataset sorted alphabetically.

        Args:
            None

        Returns:
            unique_words (List[str]): Unique words in the dataset sorted alphabetically.
        """
        #First we initialize a list that will contain the unique words
        unique_words = []

        #We iterate over all the documents in the dataset and add their word tokens to the list
        for text in self.dataset[self.COLUMN_TO_QUERY+ " (PROCESSED)"]:
            #If the document is empty, we skip it
            if text == "":
                continue
            #If the document is not empty, we add its word tokens to the list
            else:
                #Here we split the document into word tokens and add them to the list
                unique_words+=text.split(",")

        #Finally we sort the list and remove the duplicates
        unique_words = sorted(list(set(unique_words)))

        #We return the unique words
        return unique_words
    
    def __get_inverted_index(self) -> Dict[int, List[int]]:
        """
        Function that gets the inverted index of the dataset.
        
        Args:
            None

        Returns:
            inverted_index (Dict[str, List[int]]): Inverted index of the dataset.
        """
        #First we check if the inverted index has been created before
        if os.path.isfile(f"data/{self.COLUMN_TO_QUERY}_inverted_index.pickle"):
            #If it has been created before, we load it
            with open(f"data/{self.COLUMN_TO_QUERY}_inverted_index.pickle", "rb") as f:
                inverted_index = pickle.load(f)
        #If it has not been created before, we create it
        else:
            #We create the inverted index by calling the __create_inverted_index function
            inverted_index = self.__create_inverted_index()

        #We return the inverted index
        return inverted_index

    
    def __create_inverted_index(self) -> Dict[int, List[int]]:
        """
        Function that creates the inverted index of the dataset.

        Args:
            None

        Returns:
            inverted_index (Dict[str, List[int]]): Inverted index of the dataset.
        """
        #First we initialize a dictionary that will contain the inverted index
        #We choose a defaultdict because we know that the keys will be integers and the values will be lists of integers
        inverted_index = defaultdict(list)

        #Now we iterate over all the documents in the dataset
        for i, text in enumerate(self.dataset[self.COLUMN_TO_QUERY+ " (PROCESSED)"]):
            #If the document is empty, we skip it since we can't query a document that has no words
            if text == "":
                continue
            #If the document is not empty, we add its word tokens to our inverted index
            else:
                #Here we split the document into word tokens
                for word in text.split(","):
                    #Here we add the index of the document to the inverted index for each word token
                    inverted_index[self.vocabulary[word]].append(i)

        #Finally we save it to a file
        with open(f"data/{self.COLUMN_TO_QUERY}_inverted_index.pickle", "wb") as f:
            pickle.dump(inverted_index, f)

        #We return the inverted index
        return inverted_index



