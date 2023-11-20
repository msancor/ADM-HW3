from sklearn.feature_extraction.text import TfidfVectorizer
from modules.data_preprocesser import DataPreprocesser
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse._csr import csr_matrix
from typing import List, Tuple, Dict
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
import heapq
import os

"""
This module contains the SearchEngine class that implements a search engine for the dataset. The Search Engine class has the following methods:
    
    - query(query_text: str) -> pd.DataFrame: Function that queries the dataset.

"""

class SearchEngine():
    """
    Class that implements a search engine for the dataset. The Search Engine class has the following class variables:
        - COLUMN_TO_QUERY (str): Column of the dataset to query.

    """
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
            (pd.DataFrame): Results of the query sorted by index.
        """
        #First we preprocess the query
        query_text = DataPreprocesser(self.dataset).get_processed_text(query_text)

        #Now we split the query into word tokens and obtain only the unique ones
        query_text = list(set(query_text.split(",")))

        #Now we get the index of the documents that contain ALL the words in the query since it is a conjunctive query
        documents_index = self.__get_document_indexes(query_text)

        #Finally we return the results sorted by index
        return self._sorted_result(documents_index, query_text)
    
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
        documents_index = self._get_intersection(documents_index)

        #Finally we return the index of the documents that contain ALL the words in the query
        return documents_index
    
    def _get_intersection(self, documents_index: List[List[int]]) -> List[int]:
        """
        Function that gets the intersection of a list of lists.

        Args:
            documents_index (List[List[int]]): List of lists.

        Returns:
            intersection (List[int]): Intersection of the lists in the list of lists.
        """
        intersection = list(set(documents_index[0]).intersection(*documents_index))
        return intersection

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
    
    def _sorted_result(self, documents_index: List[int], query_text: List[str]) -> pd.DataFrame:
        """
        Function that sorts the results of a query.

        Args:
            documents_index (List[int]): Index of the documents that contain ALL the words in the query.
            query_text (List[str]): Query tokens.

        Returns:
            pd.DataFrame: Sorted results of a query.
        """
        #Finally we filter the dataset to get the results of the query
        result = self.dataset.iloc[documents_index][["courseName", "universityName", "description", "url"]]
        #We sort the results by index
        return result.sort_index()
    
class TopKSearchEngine(SearchEngine):
    """
    Class that implements a search engine for the dataset. This Search Engine inherits from the SearchEngine class in order to sort the results of the query by their cosine similarity with the query. 
    The Search Engine class has the following class variables:

        - COLUMN_TO_QUERY (str): Column of the dataset to query.
        - TOP_K (int): Number of results to return.
    """
    K = 5

    def __init__(self, processed_dataset: pd.DataFrame):
        """
        Function that initializes the class.

        Args:
            processed_dataset (pd.DataFrame): Processed dataset i.e. the dataset after applying the DataPreprocesser class.
        """
        #We initialize the parent class
        super().__init__(processed_dataset)
        #Here we overwrite the inverted index to be the idtf inverted index
        self.tfIdf_vectorizer = TfidfVectorizer(stop_words = [], token_pattern=r"(?u)\b\w+\b", vocabulary=self.vocabulary)
        self.tfIdf_matrix = self.__get_tfIdf_matrix()
        self.inverted_index = self.__get_tfIdf_inverted_index()

    def __get_tfIdf_inverted_index(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Function that gets the tfIdf inverted index of the dataset.
        
        Args:
            None

        Returns:
            inverted_index (Dict[int, List[Tuple[int, float]]]): Inverted index of the dataset.
        """
        #First we check if the tfIdf inverted index has been created before
        if os.path.isfile(f"data/{self.COLUMN_TO_QUERY}_tfIdf_inverted_index.pickle"):
            #If it has been created before, we load it
            with open(f"data/{self.COLUMN_TO_QUERY}_tfIdf_inverted_index.pickle", "rb") as f:
                inverted_index = pickle.load(f)
        #If it has not been created before, we create it
        else:
            #We create the tfIdf inverted index by calling the __create_tfIdf_inverted_index function
            inverted_index = self.__create_tfIdf_inverted_index()

        #We return the tfIdf inverted index
        return inverted_index
    
    def __create_tfIdf_inverted_index(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Function that creates the tfIdf inverted index of the dataset.

        Args:
            None

        Returns:
            inverted_index (Dict[int, List[Tuple[int, float]]]): Inverted index of the dataset.
        """
        #First we initialize a dictionary that will contain the tfIdf inverted index
        #We choose a defaultdict because we know that the keys will be integers and the values will be lists of tuples of integers and floats
        inverted_index = defaultdict(list)

        #Now, we get the tfIdf matrix of the dataset
        tfIdf_matrix = self.__get_tfIdf_matrix()

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
                    inverted_index[self.vocabulary[word]].append((i, tfIdf_matrix[i, self.vocabulary[word]]))

        #Finally we save it to a file
        with open(f"data/{self.COLUMN_TO_QUERY}_tfIdf_inverted_index.pickle", "wb") as f:
            pickle.dump(inverted_index, f)

        #We return the tfIdf inverted index
        return inverted_index
    
    def __get_tfIdf_matrix(self) -> np.ndarray:
        """
        Function that gets the tfIdf matrix of the dataset.
        
        Args:
            None

        Returns:
            tfIdf_matrix (np.ndarray): TfIdf matrix of the dataset.
        """
        #First we check if the tfIdf matrix has been created before
        if os.path.isfile(f"data/{self.COLUMN_TO_QUERY}_tfIdf_matrix.pickle"):
            #If it has been created before, we load it
            with open(f"data/{self.COLUMN_TO_QUERY}_tfIdf_matrix.pickle", "rb") as f:
                tfIdf_matrix = pickle.load(f)
        #If it has not been created before, we create it
        else:
            #We create the tfIdf matrix by calling the __create_tfIdf_matrix function
            tfIdf_matrix = self.__create_tfIdf_matrix()

        #We return the tfIdf matrix
        return tfIdf_matrix
    
    def __create_tfIdf_matrix(self) -> csr_matrix:
        """
        Function that creates the tfIdf matrix of the dataset.

        Args:
            None

        Returns:
            tfIdf_matrix (csr_matrix): TfIdf matrix of the dataset.
        """

        #We get the tfIdf matrix of the dataset
        tfIdf_matrix = self.tfIdf_vectorizer.fit_transform(self.dataset[self.COLUMN_TO_QUERY+ " (PROCESSED)"])

        #Finally we save it to a file
        with open(f"data/{self.COLUMN_TO_QUERY}_tfIdf_matrix.pickle", "wb") as f:
            pickle.dump(tfIdf_matrix, f)

        #We return the tfIdf matrix
        return tfIdf_matrix
    
    def _get_intersection(self, documents_index: List[List[Tuple[int, float]]]) -> List[int]:
        """
        Function that gets the intersection of a list of lists. This function is overwritten from the parent class since the inverted index includes tuples of integers and floats.

        Args:
            documents_index (List[List[Tuple[int, float]]]): List of lists.

        Returns:
            intersection (List[int]): Intersection of the lists in the list of lists.
        """
        documents_index = [[tup[0] for tup in inner_list] for inner_list in documents_index]
        #We have to take into account only the first element of the tuple i.e. the index of the document
        intersection = intersection = list(set(documents_index[0]).intersection(*documents_index))
        return intersection
    
    def _sorted_result(self, documents_index: List[int], query_text: List[str]) -> pd.DataFrame:
        """
        Function that sorts the results of a query by their cosine similarity with the query.
        It should return the top K results (if there are K, if not, the ones existing) mantained using a max heap structure.

        Args:
            documents_index (List[int]): Index of the documents that contain ALL the words in the query.
            query_text (List[str]): Query tokens.

        Returns:
            pd.DataFrame: Sorted results of a query.
        """
        #First, we get the cosine similarity between the query and the documents
        cosine_similarity = self.__get_cosine_similarity(documents_index, query_text)

        #Now we get the top K results
        topK_results, topK_sim = self.__get_top_k_results(cosine_similarity, documents_index)

        #Here we create a copy of our dataset
        result_dataset = self.dataset.copy()

        #Now we filter the dataset by our top K results
        result_dataset = result_dataset.iloc[topK_results][["courseName", "universityName", "description", "url"]]

        #Finally we add a new column with the cosine similarity values
        result_dataset["similarity"] = topK_sim

        #Finally we return the results sorted by index
        return result_dataset
    
    def __get_cosine_similarity(self, documents_index: List[int], query_text: List[str]) -> np.ndarray:
        """
        Function that gets the cosine similarity between the query and the documents.

        Args:
            documents_index (List[int]): Index of the documents that contain ALL the words in the query.
            query_text (List[str]): Query tokens.

        Returns:
            cosine_similarity (np.ndarray): Cosine similarity between the query and the documents.
        """
        #We get the tfIdf vector of the query
        query_tfIdf_vector = self.tfIdf_vectorizer.fit_transform([",".join(query_text)])

        #We get the tfIdf matrix of the documents contained in the documents_index
        tfIdf_matrix = self.tfIdf_matrix[documents_index]

        #We get the cosine similarity between the query and the documents
        similarity = cosine_similarity(query_tfIdf_vector, tfIdf_matrix)[0]

        #We return the cosine similarity
        return similarity
    
    def __get_top_k_results(self, cosine_similarity: np.ndarray, documents_index: List[int]) -> Tuple[List[int], List[float]]:

        #First we make a list of tuples where the first element is the cosine similarity and the second element is the document index
        #It is important to note that we multiply the cosine similarity values with -1 in order to build a max heap since the heapq package builds a min heap
        cosine_heap = list(zip((-1)*cosine_similarity, documents_index))
        heapq.heapify(cosine_heap)

        #Here we create a list to store the sorted top k results
        documents_list, cosine_sim_list = [], []

        for _ in range(min(self.K, len(documents_index))):
            cosine_sim, document = heapq.heappop(cosine_heap)
            cosine_sim_list.append((-1)*cosine_sim)
            documents_list.append(document)

        return documents_list, cosine_sim_list
