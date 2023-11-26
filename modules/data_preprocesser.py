#Here we import the libraries we use.
from forex_python.converter import CurrencyRates, CurrencyCodes
from typing import List, Tuple
import pandas as pd
import numpy as np
import regex as re
import datetime
import nltk

"""
This module contains the DataPreprocesser class. This class preprocesses the fees column and the text columns of the MSc courses dataset. The DataPreprocesser class has the following methods:

- preprocess_fees_column: Processes the dataframe's fees string column to a numeric column with the fees in euros. It takes the maximum value of the fees included in the fees column converted to euros.
- preprocess_text_column: Processes the dataframe's text columns to a processed text column. It tokenizes the text, removes stopwords, gets the POS tags of the tokens and lemmatizes them. At the end, it joins the lemmatized tokens into a string separated by commas.

"""

class DataPreprocesser():
    """
    Class that preprocesses the fees column and the text columns of the MSc courses dataset. The DataPreprocesser class has the following class variables:

    - CURRENCY_DATETIME: Date of when to take the exchange rate from. It is taken from the last update of the dataset.
    - STOP_WORDS: Set of stopwords from the nltk library.
    - CURRENCY_SYMBOLS: Dictionary that maps currency symbols to ISO currency codes.

    """
    #Here we define the class variables.
    #The CURRENCY_DATETIME attribute is the date of when to take the exchange rate from. It is taken from the last update of the dataset.
    CURRENCY_DATETIME = datetime.datetime(2023, 11, 15)
    #The STOP_WORDS attribute is a set of stopwords from the nltk library.
    STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
    #The CURRENCY_SYMBOLS attribute is a dictionary that maps currency symbols to ISO currency codes.
    CURRENCY_SYMBOLS = {'€': 'EUR','£': 'GBP','$': 'USD','HK$': 'HKD','¥': 'JPY', 'C$': 'CAD','A$': 'AUD', 'US$': 'USD'}

    def __init__(self, dataframe: pd.DataFrame):
        """
        Function that initializes the class.

        Args:
            dataframe (pd.DataFrame): Dataframe to preprocess.
        """
        #Here we define the attributes of the class.
        #The dataframe attribute is the dataframe to preprocess.
        self.dataset = dataframe

    def preprocess_fees_column(self, fees_column_name: str= "fees", fees_column_name_new: str= "fees (EUR)") -> None:
        """
        Function that processes the dataframe's fees string column to a numeric column with the fees in euros.

        Args:
            fees_column_name (str, optional): Name of the fees column. Defaults to "fees".
            fees_column_name_new (str, optional): Name of the new fees column. Defaults to "fees_eur".

        Returns:
            None: None.
        """

        #First, we fill the NaN values with the string "nan" to avoid errors.
        self.dataset[fees_column_name] = self.dataset[fees_column_name].fillna("nan")
        #Now, we create a new float64 column with the fees in euros by applying the __get_max_value function to the fees column.
        #The __get_max_value function returns the maximum value of the fees included in the fees column converted to euros.
        self.dataset[fees_column_name_new] = self.dataset[fees_column_name].apply(self.__get_max_value).astype(np.float64)
        
        #Finally, we return None since we are modifying the dataframe in place.
        return None

    def preprocess_text_column(self, column_name: str = "description") -> None:
        """
        Function that processes the dataframe's text column to a processed text column.

        Args:
            columns_name (str, optional): Name of the text column. Defaults to "description".

        Returns:
            None: None.
        """
        #Here we fill the NaN values with an empty string to avoid errors.
        self.dataset[column_name] = self.dataset[column_name].fillna("")
        #Now, we create a new column with the processed text by applying the __get_processed_text function to the text column.
        #The __get_processed_text function returns the processed text of the text column.
        self.dataset[f"{column_name} (PROCESSED)"] = self.dataset[column_name].apply(self.get_processed_text)

        #Finally, we return None since we are modifying the dataframe in place.
        return None
    
    def get_processed_text(self, text: str) -> str:
        """
        Function that processes a text. This function tokenizes the text, removes stopwords, gets the POS tags of the tokens and lemmatizes them.

        Args:
            text (str): Text to process.

        Returns:
            processed_text (str): Processed text.
        """
        #Here we tokenize the text (from lowercase text)
        tokens = self.__tokenizing(text)
        #Here we remove the stopwords from the tokens (and also punctuation and other non-alphanumeric characters).
        tokens_without_stopwords = self.__removing_stopwords(tokens)
        #Here we get the POS tags of the tokens (from the WordNet tagset)
        tokens_with_wordnet_tags = self.__get_POS_tags(tokens_without_stopwords)
        #Here we lemmatize the tokens (using the WordNet lemmatizer)
        lemmatized_tokens = self.__lemmatizing(tokens_with_wordnet_tags)
        #Here we join the lemmatized tokens into a string separated by commas.
        processed_text = ",".join(lemmatized_tokens)
        
        #Here we return the processed text.
        return processed_text
    
    def __get_max_value(self, fees_text:str) -> float:
        """
        Function that returns the maximum value of the fees included in a string converted to euros.

        Args:
            fees_text (str): String to extract the fees from.

        Returns:
            final_value (float): Maximum value of the fees included in the string converted to euros.
        """
        #First, we initialize the list of converted values to store the converted values of the fees (to euros).
        converted_values = []
        #Now, we find all the fees in the string and store them in a list of tuples. Each tuple contains the value and the currency of the fee.
        fees_list = self.__find_currency_strings(fees_text)

        #The first filter is checking if the fees_list is empty. If it is, we return NaN since we cannot extract any fees from the string.
        if len(fees_list) == 0:
            return np.nan
        
        #Now, we loop through the list of tuples to convert the values to euros and the symbols to ISO currency codes.
        for value, currency in fees_list:
            #First, we parse the currency to its ISO code if it is a symbol and if it is not we leave it as it is.
            currency = self.__parse_currency(currency)
            #Now, we parse the value to a float.
            value = self.__parse_value(value)
            #Finally, we convert the value to euros.
            value = self.__convert_to_euros(value, currency)
            #Here we append the converted value to the list of converted values.
            converted_values.append(value)
        
        #The second filter is checking if the list of converted values is empty. If it is, we return NaN since we cannot extract any fees from the string.
        if len(converted_values) == 0:
            return np.nan
        
        #Finally, we obtain the maximum value of the converted values rounded to 3 decimals.
        final_value = round(max(converted_values),3)

        #Here we return the maximum value of the converted values rounded to 3 decimals.
        return final_value

    def __find_currency_strings(self, fees_text: str) -> List[Tuple[str, str]]:
        """
        Function that finds all the fees in a string and returns them in a list of tuples. Each tuple contains the value and the currency of the fee.
        We use a Regex to find the fees in the string. The Regex has the following structure:
        - (\p{Sc}) finds currency symbols and stores them in a group.
        - \s? finds one or zero spaces.
        - (\d+[\.\,\s]{0,1}[\d\.\,]{0,}) finds one or more digits followed by zero or one dot, comma or space followed by zero or more digits or dots or commas and stores them in the second group.
        - (A-Z{3}) finds three capital letters and stores them in a group. This is to find the ISO currency codes.
        - ([A-Z\p{Sc}]{3}) finds three capital letters or currency symbols and stores them in a group. This is to find the ISO currency codes or the currency symbols.

        Args:
            fees_text (str): String to extract the fees from.

        Returns:
            fees_list (List[Tuple[str, str]]): List of tuples with the fees and their currencies.
        """
        #We initialize the list of tuples to store the fees and their currencies.
        fees_list = []

        #First we pass a filter to find all the fees that are in the format of € 2,209, €2,209, etc.
        first_filter = self.__sorted_group_search(fees_text, r'(\p{Sc})\s?(\d+[\.\,\s]{0,1}[\d\.\,]{0,})', switch=True)
        #Now, we pass a filter to find all the fees that are in the format of 2,209 €, 2,209€, etc.
        second_filter = self.__sorted_group_search(fees_text, r'(\d+[\.\,\s]{0,1}[\d\.\,]{0,})\s?(\p{Sc})')
        #Here we pass a filter to find all the fees that are in the format of EUR 2,209, EUR2,209, etc.
        third_filter = self.__sorted_group_search(fees_text, r'([A-Z\p{Sc}]{3})\s?(\d+[\.\,\s]{0,1}[\d\.\,]{0,})', switch=True)
        #Now, we pass a filter to find all the fees that are in the format of 2,209 EUR, 2,209EUR, etc.
        fourth_filter = self.__sorted_group_search(fees_text, r'(\d+[\.\,\s]{0,1}[\d\.\,]{0,})\s?([A-Z\p{Sc}]{3})')
        #Here we pass a filter to find all the fees that are in the format of € 2,209 EUR, €2,209EUR, etc. In this case we only take the ISO currency code and the value
        fifth_filter = self.__sorted_group_search(fees_text, r'\p{Sc}\s?(\d+[\.\,\s]{0,1}[\d\.\,]{0,})\s?([A-Z]{3})')
        #Finally, we pass a filter to find all the fees that are EUR € 2,209, EUR€2,209, etc. In this case we only take the ISO currency code and the value
        sixth_filter = self.__sorted_group_search(fees_text, r'([A-Z]{3})\s?\p{Sc}\s?(\d+[\.\,\s]{0,1}[\d\.\,]{0,})', switch=True)
        
        #Here we extend the list of tuples with the fees and their currencies with the results of the filters.
        fees_list.extend([*first_filter, *second_filter, *third_filter, *fourth_filter, *fifth_filter, *sixth_filter])

        #Finally, we sort and delete duplicates from the list of tuples by enforcing that the first element of the tuple (the value) is unique and we mantain the ISO currency code if it is present.
        #We also enforce that symbols are first if they are present.
        fees_list = self.__sort_and_delete_duplicates(fees_list)

        #Here we return the list of tuples with the fees and their currencies.
        return fees_list
    
    
    def __parse_currency(self, currency: str) -> str:
        """
        Function that parses a currency string to its ISO code.

        Args:
            currency (str): Currency string to parse.

        Returns:
            currency (str): ISO code of the currency.
        """
        #First we check if the currency is a symbol. If it is, we parse it to its ISO code.
        if not currency.isalpha():
            try:
                #Here we parse the currency to its ISO code by checking if it is in the symbols dictionary.
                currency = self.CURRENCY_SYMBOLS[currency]
            except:
                #If the currency is not in the symbols dictionary, we raise an exception.
                raise Exception(f"Currency symbol {currency} not found in the symbols dictionary")
        
        #If the currency is not a symbol, we return it as it is since it is already an ISO code.
        return currency
    
    def __parse_value(self, value: str) -> float:
        """
        Function that parses a value string to a float. This function removes the thousands separator and the decimal separator.
        We use a Regex to parse the value string. The Regex has the following structure:
        - (\d+) finds one or more digits and stores them in a group.
        - [\.\,\s]? finds one or zero dots, commas or spaces.
        - (\d+)? finds zero or one repetitions of one or more digits and stores them in a group.
        - [\,]? finds zero or one commas.

        Args:
            value (str): Value string to parse.

        Returns:
            value (float): Parsed value.
        """
        #Here we parse the value string to a float by removing the thousands separator and the decimal separator using the re.sub function.
        #This function substitutes the matches of the Regex with the second argument of the function.
        value = float(re.sub(r'(\d+)[\.\,\s]?(\d+)?[\,]?', r'\1\2', value))

        #Here we return the parsed value.
        return value
    
    def __convert_to_euros(self, value: float, currency: str) -> float:
        """
        Function that converts a value from a given currency to euros.

        Args:
            value (float): Value to convert.
            currency (str): ISO code of the currency to convert from.

        Returns:
            value (float): Converted value.
        """
        #Here we convert the value from the given currency to euros using the CurrencyRates().convert function.
        #This function takes the value, the currency to convert from, the currency to convert to and the date of when to take the exchange rate from.
        #In this case, we take the exchange rate from the date of the last update of the dataset. The date is stored in the CURRENCY_DATETIME attribute.
        value = CurrencyRates().convert(currency, 'EUR', value, self.CURRENCY_DATETIME)

        #Here we return the converted value.
        return value
    
    def __sorted_group_search(self, text: str, regex: str, switch: bool = False) -> List[Tuple[str, str]]:
        """
        Function that searches for a regex in a text and returns the matches reordering the groups of the regex.
        This function is used to find the fees in a string. The fees can be in different formats, so we use different regexes to find them.
        We also validate via this function that all the currency values are effectively currencies and not just words.

        Args:
            text (str): Text to search in.
            regex (str): Regex to search for.
            switch (bool, optional): Boolean to switch the order of the groups of the regex. Defaults to False.

        Returns:
            List[Tuple[str, str]]: List of tuples with the matches of the regex.

        """
        #If the switch is True, we switch the order of the groups of the regex.
        if switch == True:
            #Here we return a list of tuples with the matches of the regex. The first element of the tuple is the second group of the regex and the second element is the first group of the regex.
            #We only return the matches that are effectively currencies by using the __validate_currency function.
            return [(value, currency) for currency, value in re.findall(regex, text) if self.__validate_currency(currency) == True]

        #If the switch is False, we do not switch the order of the groups of the regex.
        return [(value, currency) for value, currency in re.findall(regex, text) if self.__validate_currency(currency) == True]
    
    def __validate_currency(self, currency: str) -> bool:
        """
        Function that validates a currency string.

        Args:
            currency (str): Currency string to validate.

        Returns:
            bool: True if the currency is valid, False otherwise.
        """
        #First we check if the currency is a symbol.
        if not currency.isalpha():
            #Here we return True since all symbols are valid currencies.
            return True
        #If the currency is not a symbol, we check if it is a valid ISO code.
        else:
            #If we can get the symbol of the currency, it is a valid ISO code. We do this by using the CurrencyCodes().get_symbol function.
            if CurrencyCodes().get_symbol(currency) is not None:
                #Here we return True since the currency is a valid ISO code.
                return True
            #If we cannot get the symbol of the currency, it is not a valid ISO code.
            else:
                #Here we return False since the currency is not a valid ISO code.
                return False
    
    def __sort_and_delete_duplicates(self, fees_list: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Function that sorts a list of tuples and deletes duplicates.

        Args:
            fees_list (List[Tuple[str, str]]): List of tuples to sort and delete duplicates from.

        Returns:
            fees_list (List[Tuple[str, str]]): Sorted list of tuples without duplicates.
        """
        #First we sort the list of tuples by the length of the second element (the currency).
        #This is to enforce that symbols are first if they are present.
        fees_list = sorted(fees_list, key=lambda x: len(x[1]))

        #Now we elminate duplicates on the first element (the value)
        #This also enforces that we keep the ISO currency code if it is present.
        fees_list = list(dict(fees_list).items())

        #Here we return the sorted list of tuples without duplicates.
        return fees_list
    
    def __tokenizing(self, text: str) -> List[str]:
        """
        Function that tokenizes a text. Tokenizing is the process of splitting a text into tokens i.e. individual words.

        Args:
            text (str): Text to tokenize.

        Returns:
            tokens (List[str]): List of tokens.
        """
        #Here we tokenize the text after converting it to lowercase.
        tokens = nltk.tokenize.word_tokenize(text.lower())
        
        #Here we return the tokens.
        return tokens
    
    def __removing_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Function that removes stopwords from a list of tokens. Stopwords are words that are not relevant for the search.
        This can include words like "the", "a", "an", etc.

        Args:
            tokens (List[str]): List of tokens to remove stopwords from.

        Returns:
            tokens_without_stopwords (List[str]): List of tokens without stopwords.
        """
        #Here we remove the stopwords from the list of tokens. We do this by checking if the token is in the STOP_WORDS set.
        #We also check if the token is alphanumeric since we want to remove punctuation and other non-alphanumeric characters like "!" or "?".
        tokens_without_stopwords =[token for token in tokens if not token in self.STOP_WORDS and token.isalnum()]

        #Here we return the tokens without stopwords.
        return tokens_without_stopwords

    def __get_POS_tags(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Function that returns the WordNet POS tags of a list of tokens. WordNet is a lexical database for the English language.
        This database groups words into sets of synonyms called synsets and describes semantic relationships between them.
        POS tags are used to indicate the part of speech of a word. They can be nouns, verbs, adjectives, adverbs, etc.

        Args:
            tokens (List[str]): List of tokens to get POS tags from.

        Returns:
            tokens_with_wordnet_tags (List[str]): List of POS tags.
        """
        #Here we get the POS tags of the tokens. These tags are given according to the Penn Treebank tagset.
        #The Penn Treebank tagset is a tagset for annotating words with their part-of-speech
        #Among other things, it distinguishes between nouns, verbs, adjectives, adverbs, etc.
        tokens_with_treebank_tags = nltk.pos_tag(tokens)

        #Now we convert the Penn Treebank tags to WordNet tags.
        tokens_with_wordnet_tags = [(token, self.__get_wordnet_tag(tag)) for (token, tag) in tokens_with_treebank_tags]

        #Then we return the tokens with WordNet tags.
        return tokens_with_wordnet_tags
    
    
    def __get_wordnet_tag(self, treebank_tag: str) -> str:
        """
        Function that converts a Penn Treebank tag to a WordNet tag. Wordnet classifies words into:
        - Nouns (n)
        - Verbs (v)
        - Adjectives (a)
        - Adverbs (r)

        Whereas the Penn Treebank tagset classifies words into:
        - Nouns (NN)
        - Verbs (VB)
        - Adjectives (JJ)
        - Adverbs (RB)

        etc. (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

        NB: Part of this function was not written by me. It was taken and modified from the following StackOverflow answer:
        https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python

        This was done because it is a very common solution to convert Penn Treebank tags to WordNet tags 
        and it is used in many solutions. We thought it was not worth writing a new solution for this.

        I don't have the rights to this code and I am not claiming it as my own.

        Args:
            treebank_tag (str): Penn Treebank tag.

        Returns:
            str: WordNet tag.
        """
        #Here we convert the Penn Treebank tag to a WordNet tag.

        #First we check if the Penn Treebank tag starts with one of the following letters:
        # - J: Adjective
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        # - V: Verb
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        # - N: Noun
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        # - R: Adverb
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        #If the Penn Treebank tag does not start with one of the previous letters, we return the default Wordnet lemmatizer tag.
        #This is the "noun" tag.
        else:
            return "n"
        
    def __lemmatizing(self, tokens_with_wordnet_tags: List[Tuple[str, str]]) -> List[str]:
        """
        Function that lemmatizes a list of tokens. Lemmatizing is the process of grouping together the inflected forms of a word so they can be analysed as a single item.
        An example of this is the word "studies". The inflected forms of this word are "study", "studies", "studying" and "studied". Lemmatizing this word would return "study".

        Args:
            tokens_with_wordnet_tags (List[Tuple[str, str]]): List of tokens with WordNet tags.

        Returns:
            List[str]: List of lemmatized tokens.
        """
        #First we initialize the WordNet lemmatizer.
        lemmatizer = nltk.stem.WordNetLemmatizer()
        #Here we lemmatize the tokens.
        lemmatized_tokens = [lemmatizer.lemmatize(token, tag) for token, tag in tokens_with_wordnet_tags]
        
        #Here we return the lemmatized tokens.
        return lemmatized_tokens
    