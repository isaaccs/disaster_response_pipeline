import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
     """loads the specified message and category data
    Args:
        messages_filepath (string): The file path of the messages csv
        categories_filepath (string): The file path of the categories cv
    Returns:
        df (pandas dataframe): The combined messages and categories df
    """
     messages = pd.read_csv(messages_filepath)
     categories = pd.read_csv(categories_filepath)
     return pd.merge(messages, categories, on='id')


def clean_data(df):
    """Cleans the data:
        - drops duplicates
        - cleans up the categories column
    Args:
        df (pandas dataframe): combined categories and messages df
    Returns:
        df (pandas dataframe): Cleaned dataframe with split categories
    """
     #Split categories into separate category columns.
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories[:1]
    #extract a list of new column names for categories.
    category_colnames = row.applymap(lambda s: s[:-2]).iloc[0, :].tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Convert category values to just numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] =categories[column].str.split('-').str[-1]
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
    #Replace categories column in df with new category columns
    df=df.drop(columns="categories")
    df = pd.concat([df, categories], axis=1)
    #Remove duplicates
    df.drop_duplicates(subset='id', inplace=True)
    return df


def save_data(df, database_filename):
    """Saves the resulting data to a sqlite db
    Args:
        df (pandas dataframe): The cleaned dataframe
        database_filename (string): the file path to save the db
    Returns:
        None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('data_cleaned', engine, index=False, if_exists='replace')
    engine.dispose()  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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