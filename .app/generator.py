import pandas as pd
from faker import Faker
import random

# Initialize Faker
fake = Faker()

def generate_synthetic_data(schema: dict, num_rows: int = 100) -> pd.DataFrame:
    """
    Generates synthetic tabular data based on a given schema.

    Args:
        schema (dict): A dictionary defining the columns and their types/generation rules.
                       Example: {'name': 'name', 'email': 'email', 'age': 'int', 'city': ['New York', 'London', 'Paris']}
        num_rows (int): The number of rows to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic data.
    """
    data = {}
    for col_name, col_type in schema.items():
        if isinstance(col_type, str): # Faker methods or basic types
            if col_type == 'name':
                data[col_name] = [fake.name() for _ in range(num_rows)]
            elif col_type == 'email':
                data[col_name] = [fake.email() for _ in range(num_rows)]
            elif col_type == 'address':
                data[col_name] = [fake.address() for _ in range(num_rows)]
            elif col_type == 'phone_number':
                data[col_name] = [fake.phone_number() for _ in range(num_rows)]
            elif col_type == 'date':
                data[col_name] = [fake.date_between(start_date='-10y', end_date='today') for _ in range(num_rows)]
            elif col_type == 'int':
                # Generate random integers between 18 and 65 (example range)
                data[col_name] = [random.randint(18, 65) for _ in range(num_rows)]
            elif col_type == 'float':
                # Generate random floats between 0.0 and 100.0 (example range)
                data[col_name] = [random.uniform(0.0, 100.0) for _ in range(num_rows)]
            elif col_type == 'sentence':
                data[col_name] = [fake.sentence() for _ in range(num_rows)]
            else: # Treat as a literal string or unknown type, can expand later
                data[col_name] = [col_type for _ in range(num_rows)] # Just repeat the string
        elif isinstance(col_type, list): # Categorical values
            data[col_name] = [random.choice(col_type) for _ in range(num_rows)]
        # Add more sophisticated types/rules here later

    return pd.DataFrame(data)

if __name__ == '__main__':
    # Example Schema for testing
    sample_schema = {
        'User_ID': 'int',
        'Customer_Name': 'name',
        'Customer_Email': 'email',
        'Registration_Date': 'date',
        'Country': ['USA', 'Canada', 'UK', 'Australia', 'Germany'],
        'Age': 'int',
        'Product_Category': ['Electronics', 'Books', 'Clothing', 'Home Goods'],
        'Price': 'float',
        'Description': 'sentence'
    }

    print("Generating 5 rows of synthetic data...")
    df = generate_synthetic_data(sample_schema, num_rows=5)
    print(df.head())

    print("\nGenerating 100 rows and saving to synthetic_data.csv...")
    df_large = generate_synthetic_data(sample_schema, num_rows=100)
    df_large.to_csv('synthetic_data.csv', index=False)
    print("Saved to synthetic_data.csv")