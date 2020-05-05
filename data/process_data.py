# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        Args:
            messages_filepath: str
                 Path to message dataset csv

            categories_filepath:str
                 Path to categories dataset csv

        Returns: list
            List of the two loaded dataframes

    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages, categories



def clean_data(df1, df2):
    """
        Args:
            df1: DataFrame
            df2: DataFrame

        Returns: DataFrame
            Merged df1 & df1, transformed, merged and returned.
    """

    df = pd.merge(df1, df2, how='inner', on='id')

    # create a dataframe of the 36 individual category columns
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # rename the columns of `categories`
    categories = df.categories.str.split(';', expand=True)
    category_colnames = [text[0:-2] for text in categories[0:1].values[0]]
    categories.columns = category_colnames


    # Convert category values to just numbers 0 or 1
    # set each value to be the last character of the string and convert column from string to numeric
    for column in categories:       
        categories[column] = list(np.vectorize(lambda text: int(text[-1]))(list(categories[column])))
    

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df.drop(columns=['categories']), categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates(keep=False)

    return df


def save_data(df, database_filename):
    """
        Args:
            df: DataFrame
                cleaned dataframe
            
            database_filename: str
                destination database path
            
        Returns: Not applicable
            
    """
    engine = create_engine(database_filename)
    df.to_sql('DisasterMessage', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df_msg, df_catig = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df_msg, df_catig)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()