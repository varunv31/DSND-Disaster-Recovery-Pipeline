import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    # Load categories.csv into a dataframe 
    categories = pd.read_csv(categories_filepath)
    # Merge Data set
    df = messages.merge(categories, how = 'left', on = ['id'])
    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(map(lambda i: i[ : -2], row)) 
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = categories[column].astype('int')
        # Replace categories column in df with new category columns.
    df.drop(['categories'],axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # replacing 2 value with 1 value
    df.replace(2, 1, inplace=True)
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('Disaster_Combined', engine, index=False)


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