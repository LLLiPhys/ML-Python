import pandas as pd

# Import module(s) for data splitting (train and test)
from sklearn.model_selection import train_test_split

# Import module(s) for data preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# Import module(s) for model building
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Import module(s) for performance scoring
from sklearn.metrics import mean_absolute_error

# from sklearn_pandas import CategoricalImputer

import warnings
warnings.filterwarnings("ignore")


# Define a function to deal with Missing Values in both train and test subsets using different strategies
def deal_with_missing_values(X_train, X_test, columns_with_missing, categorical_columns, numerical_columns, strategy="simple_dropping"):
    """
    Args:
        strategy (str): "simple_dropping" or "simple_imputing.
    """

    # The Approach 1: Drop those columns with Missing Values
    if strategy == "simple_dropping":
        X_train_new = X_train.drop(columns_with_missing, axis=1)
        X_test_new = X_test.drop(columns_with_missing, axis=1)
        # print(X_train_new.head(5))
        
    # The Approach 2: Fill in Missing Values with Imputation
    if strategy == "simple_imputing":
        simple_imputer = SimpleImputer()
        # Exclude categorical columns to keep only numerical columns
        X_train_numerical = X_train[numerical_columns].copy()
        X_test_numerical =  X_test[numerical_columns].copy()
        # Make a simple numerical imputation using fit_transform method
        X_train_numerical_imputed = pd.DataFrame(simple_imputer.fit_transform(X_train_numerical))
        X_test_numerical_imputed = pd.DataFrame(simple_imputer.transform(X_test_numerical))
        # Imputation removed column names and thus put them back
        X_train_numerical_imputed.columns = X_train_numerical.columns
        X_test_numerical_imputed.columns = X_test_numerical.columns
        X_train_numerical_imputed.index = X_train.index
        X_test_numerical_imputed.index = X_test.index
        # print(X_train_numerical_imputed.head(5))
        
        # Extract categorical columns for later combination with numerical columns
        X_train_categorical = X_train[categorical_columns].copy()
        X_test_categorical = X_test[categorical_columns].copy()
        
        # Fill in possible NaN values with most frequent values
        X_train_categorical = X_train_categorical.apply(lambda x: x.fillna(x.value_counts().index[0]))
        X_test_categorical = X_test_categorical.apply(lambda x: x.fillna(x.value_counts().index[0]))

        # Make a simple categorical imputation
        # category_imputer = CategoricalImputer()
        # X_train_categorical_imputed = pd.DataFrame(category_imputer.fit_transform(X_train_categorical))
        # X_test_categorical_imputed = pd.DataFrame(category_imputer.fit_transform(X_test_categorical))
        
        # Merge numerical columns with categorical columns
        X_train_new = pd.concat([X_train_numerical_imputed, X_train_categorical], axis=1)
        X_test_new = pd.concat([X_test_numerical_imputed, X_test_categorical], axis=1)
   
    return X_train_new, X_test_new
    
# Define a function to deal with Categorical Variables in both train and test subsets using different strategies    
def deal_with_categorical_variables(X_train, X_test, columns_with_missing, categorical_columns, numerical_columns, strategy="simple_dropping"):
    """
    Args:
        strategy (str): "simple_dropping" or "ordinal_encoding" or "onehot_encoding".
    """    

    # Merge selected categorical and numerical columns
    merged_columns = categorical_columns + numerical_columns
    X_train_copy = X_train[merged_columns].copy()
    X_test_copy = X_test[merged_columns].copy()

    # The Approach 1: Simply Dropping Categorical Variables
    if strategy == "simple_dropping":
        X_train_new = X_train_copy.select_dtypes(exclude=['object'])
        X_test_new = X_test_copy.select_dtypes(exclude=['object'])

    # The Approach 2: Ordinally Encode Categorical Variables
    if strategy == "ordinal_encoding":
        X_train_new = X_train_copy.copy()
        X_test_new = X_test_copy.copy()
        # Apply ordinal encoder to each column with categorical data
        ordinal_encoder = OrdinalEncoder()
        X_train_new[categorical_columns] = ordinal_encoder.fit_transform(X_train_copy[categorical_columns])
        X_test_new[categorical_columns] = ordinal_encoder.transform(X_test_copy[categorical_columns])

    # The Approach 3: One-Hot Encode Categorical Variables
    if strategy == "onehot_encoding":
        # Apply one-hot encoder to each column with categorical data
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_train_categorical = pd.DataFrame(onehot_encoder.fit_transform(X_train_copy[categorical_columns]))
        X_test_categorical = pd.DataFrame(onehot_encoder.transform(X_test_copy[categorical_columns]))
        # One-hot encoding removed index; put it back
        X_train_categorical.index = X_train_copy.index
        X_test_categorical.index = X_test_copy.index
        # Extract numerical columns
        # X_train_numerical = X_train_copy.drop(categorical_columns, axis=1)
        # X_test_numerical = X_test_copy.drop(categorical_columns, axis=1)
        X_train_numerical = X_train_copy[numerical_columns].copy()
        X_test_numerical =  X_test_copy[numerical_columns].copy()
        # Merge categorical and numerical columns
        X_train_new = pd.concat([X_train_numerical, X_train_categorical], axis=1)
        X_test_new = pd.concat([X_test_numerical, X_test_categorical], axis=1)    
    
    return X_train_new, X_test_new

# Define a function to measure the performance of each approach
def score_model(X_train, X_test, y_train, y_test, model="RFG", **kwargs):
    
    if model == "RandomForestRegressor":
        model = RandomForestRegressor(**kwargs)
    if model == "LinearRegression":
        model = LinearRegression(**kwargs)
    if model == "LogisticRegression":
        model = LogisticRegression(**kwargs)
        
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    return mean_absolute_error(y_test, preds)


def main():
    
    # Load the dataset
    data = pd.read_csv("melb_data.csv")

    # Separate target from predictors
    y = data["Price"]
    X = data.drop(["Price"], axis=1)
    # X = data.select_dtypes(exclude=["object"])

    # Divide data into training and validation subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    columns_with_missing = [column for column in X_train.columns if X_train[column].isnull().any()]
    print("Columns with Missing values:")
    print(columns_with_missing)

    missing_values_by_column = (X_train.isna().sum()) # X_train.isnull().sum()
    print("Number of Missing Values by Column:")
    print(missing_values_by_column)
    print(missing_values_by_column[missing_values_by_column>0])

    # Select categorical columns 
    max_unique = None
    if max_unique is None:
        categorical_columns = [column for column in X_train.columns if X_train[column].dtype == "object"]   
    else:     
        categorical_columns = [column for column in X_train.columns if X_train[column].nunique() < max_unique and X_train[column].dtype == "object"]
    print("Categorical Columns:")
    print(categorical_columns)

    # Select numerical columns
    numerical_columns = [column for column in X_train.columns if X_train[column].dtype != "object"]
    print("Numerical Columns:")
    print(numerical_columns)


    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    X_train_new, X_test_new = deal_with_missing_values(X_train_new, X_test_new, columns_with_missing, categorical_columns, numerical_columns, strategy="simple_imputing")
    X_train_new, X_test_new = deal_with_categorical_variables(X_train_new, X_test_new, columns_with_missing, categorical_columns, numerical_columns, strategy="onehot_encoding")
    model_score = score_model(X_train_new, X_test_new, y_train, y_test, model="RandomForestRegressor", n_estimators=100, random_state=0)
    print("Model Score:", model_score)
    
    # for strategy1 in ["simple_dropping", "simple_imputing"]:
    #     for strategy2 in ["simple_dropping", "ordinal_encoding"]:
    #         X_train_new = X_train.copy()
    #         X_test_new = X_test.copy()
    #         X_train_new, X_test_new = deal_with_missing_values(X_train_new, X_test_new, strategy=strategy1)
    #         X_train_new, X_test_new = deal_with_categorical_variables(X_train_new, X_test_new, strategy=strategy2)
    #         print(f"MAE from the Approaches {strategy1} + {strategy2}")
    #         print(score_model(X_train_new, X_test_new, y_train, y_test))
    
if __name__ == "__main__":
    main()
