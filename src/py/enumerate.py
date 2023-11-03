import pandas as pd
import argparse
def enumerate_labels(df, input_col, output_col):
    """
    Enumerates unique labels in the input column and assigns them an integer ID in the output column.

    Parameters:
    - df (pd.DataFrame): Input pandas dataframe
    - input_col (str): Name of the column containing labels
    - output_col (str): Name of the new column where the integer IDs will be stored

    Returns:
    - pd.DataFrame: DataFrame with new column added containing enumerated IDs
    """
    # Assign a unique integer ID to each unique label in the input column
    df[output_col] = df[input_col].astype('category').cat.codes
    return df

def main(args):

    enumerate_labels(pd.read_csv(args.csv_train), args.class_column, args.output_column).to_csv(args.csv_train, index=False)
    enumerate_labels(pd.read_csv(args.csv_valid), args.class_column, args.output_column).to_csv(args.csv_valid, index=False)
    enumerate_labels(pd.read_csv(args.csv_test), args.class_column, args.output_column).to_csv(args.csv_test, index=False)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Enumerate')
    parser.add_argument('--csv_train', help='CSV for training', type=str, required=True)
    parser.add_argument('--csv_valid', help='CSV for validation', type=str, required=True)
    parser.add_argument('--csv_test', help='CSV for testing', type=str, required=True)
    parser.add_argument('--img_column', help='Image/video column name', type=str, default='vid_path')
    parser.add_argument('--class_column', help='Class column', type=str, default='class')
    parser.add_argument('--output_column', help='Class column', type=str, default='class')

    args = parser.parse_args()

    main(args)
