import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sqlalchemy import create_engine, text
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.layouts import gridplot

# Function to calculate the sum of squared deviations
def sum_squared_deviations(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# Function to find the best fit ideal functions
def find_best_fit_functions(train_data, ideal_data):
    num_ideal_functions = ideal_data.shape[1] - 1
    best_fit_functions = []
    
    for i in range(1, num_ideal_functions + 1):
        x_train = train_data[:, 0]
        y_train = train_data[:, 1]
        
        x_ideal = ideal_data[:, 0]
        y_ideal = ideal_data[:, i]
        
        coefficients = np.polyfit(x_train, y_train, 1)  # Linear regression
        y_pred = np.polyval(coefficients, x_ideal)
        deviation = sum_squared_deviations(y_ideal, y_pred)
        
        best_fit_functions.append((deviation, coefficients))
    
    # Sort the best fit functions based on deviation
    best_fit_functions.sort(key=lambda x: x[0])
    
    return best_fit_functions[:4]  # Return top 4 best fit functions

# Function to map test data to the chosen ideal functions
def map_test_data(test_data, chosen_functions):
    mappings = []
    
    for row in test_data:
        x_test, y_test = row
        assigned_functions = []
        
        for deviation, coefficients in chosen_functions:
            y_pred = np.polyval(coefficients, x_test)
            
            if deviation <= np.max(test_data[:, 1]) - np.min(test_data[:, 1]) * np.sqrt(2):
                assigned_functions.append((y_pred, deviation))
        
        mappings.append(assigned_functions)
    
    return mappings

# Read training data from CSV file
with open('train.csv', 'r') as file:  # Replace 'train.csv' with your training data file name
    reader = csv.reader(file)
    next(reader)  # Skip header row
    train_data = np.array([list(map(float, row)) for row in reader])

# Read test data from CSV file
with open('test.csv', 'r') as file:  # Replace 'test.csv' with your test data file name
    reader = csv.reader(file)
    next(reader)  # Skip header row
    test_data = np.array([list(map(float, row)) for row in reader])

# Read ideal functions data from CSV file
with open('ideal.csv', 'r') as file:  # Replace 'ideal.csv' with your ideal functions file name
    reader = csv.reader(file)
    next(reader)  # Skip header row
    ideal_data = np.array([list(map(float, row)) for row in reader])

# Find the best fit ideal functions
chosen_functions = find_best_fit_functions(train_data, ideal_data)

# Map test data to the chosen ideal functions
mappings = map_test_data(test_data, chosen_functions)

# Visualize the data
plt.scatter(train_data[:, 0], train_data[:, 1], color='blue', label='Training Data')
plt.scatter(test_data[:, 0], test_data[:, 1], color='red', label='Test Data')

for i, (deviation, coefficients) in enumerate(chosen_functions):
    x_ideal = ideal_data[:, 0]
    y_ideal = np.polyval(coefficients, x_ideal)
    plt.plot(x_ideal, y_ideal, label=f'Ideal Function {i+1}')
    
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#Display the coefficents  and deviation of functions

for i, (deviation, coefficients) in enumerate(chosen_functions):
    print(f"Ideal Function {i+1}:")
    print(f"Deviation: {deviation}")
    print(f"Coefficients: {coefficients}\n")


# Extract equations for chosen ideal functions

equations = []
for i, (_, coefficients) in enumerate(chosen_functions):
    slope, intercept = coefficients
    equation = f"y = {slope:.2f}x + {intercept:.2f}"
    equations.append(equation)
    print(f"Ideal Function {i+1}: {equation}")


def create_database(database_file):
    engine = create_engine(f'sqlite:///{database_file}')
    conn = engine.connect()

    # Create tables in the database
    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS training_data (
            x REAL,
            y1 REAL,
            y2 REAL,
            y3 REAL,
            y4 REAL
        )
    '''))

    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS ideal_functions (
            x REAL,
            y1 REAL,
            y2 REAL,
            y3 REAL,
            y4 REAL,
            y5 REAL,
            y6 REAL,
            y7 REAL,
            y8 REAL,
            y9 REAL,
            y10 REAL,
            y11 REAL,
            y12 REAL,
            y13 REAL,
            y14 REAL,
            y15 REAL,
            y16 REAL,
            y17 REAL,
            y18 REAL,
            y19 REAL,
            y20 REAL,
            y21 REAL,
            y22 REAL,
            y23 REAL,
            y24 REAL,
            y25 REAL,
            y26 REAL,
            y27 REAL,
            y28 REAL,
            y29 REAL,
            y30 REAL,
            y31 REAL,
            y32 REAL,
            y33 REAL,
            y34 REAL,
            y35 REAL,
            y36 REAL,
            y37 REAL,
            y38 REAL,
            y39 REAL,
            y40 REAL,
            y41 REAL,
            y42 REAL,
            y43 REAL,
            y44 REAL,
            y45 REAL,
            y46 REAL,
            y47 REAL,
            y48 REAL,
            y49 REAL,
            y50 REAL
        )
    '''))

    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS test_data (
            x REAL,
            y REAL
        )
    '''))

    conn.execute(text('''
        CREATE TABLE IF NOT EXISTS results (
            x REAL,
            y REAL,
            chosen_function INTEGER,
            deviation REAL
        )
    '''))

    conn.close()

def insert_data_to_table(data, table_name, database_file):
    engine = create_engine(f'sqlite:///{database_file}')
    conn = engine.connect()
    data.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

def main(train_file, ideal_file, test_file, database_file):
    # Create the database
    create_database(database_file)

    # Read the data from CSV files
    train_data = pd.read_csv(train_file)
    ideal_data = pd.read_csv(ideal_file)
    test_data = pd.read_csv(test_file)

    # Insert the data into the database tables
    insert_data_to_table(train_data, 'training_data', database_file)
    insert_data_to_table(ideal_data, 'ideal_functions', database_file)
    insert_data_to_table(test_data, 'test_data', database_file)

    # Perform further processing and analysis

    # ...

    # Example: Query data from the database
    engine = create_engine(f'sqlite:///{database_file}')
    conn = engine.connect()
    result = conn.execute(text('SELECT * FROM training_data')).fetchall()
    #print(result)
    conn.close()

# Specify the file names
train_file = 'train.csv'
ideal_file = 'ideal.csv'
test_file = 'test.csv'
database_file = 'data.db'

# Run the main function
main(train_file, ideal_file, test_file, database_file)

def find_best_fit_functions(train_data, ideal_data):
    num_ideal_functions = ideal_data.shape[1] - 1
    chosen_functions = []

    for i in range(num_ideal_functions):
        x_train = train_data[:, 0]
        y_train = train_data[:, i + 1]
        x_ideal = ideal_data[:, 0]
        y_ideal = ideal_data[:, i + 1]

        # Fit a polynomial of degree 1
        coefficients = np.polyfit(x_train, y_train, deg=1)
        fitted_values = np.polyval(coefficients, x_train)

        # Calculate deviation as the absolute difference between fitted values and ideal values
        deviation = np.abs(fitted_values - y_ideal)

        # Visualize the deviation
        plt.figure()
        plt.plot(x_train, deviation)
        plt.xlabel('x')
        plt.ylabel('Deviation')
        plt.title(f'Ideal Function {i+1} Deviation')
        plt.show()

        # Find the index with the minimum deviation
        chosen_function_index = np.argmin(deviation)
        chosen_functions.append(chosen_function_index)

    return chosen_functions

def main(train_file, ideal_file, test_file, database_file):
    train_data = pd.read_csv(train_file)
    ideal_data = pd.read_csv(ideal_file)
    test_data = pd.read_csv(test_file)

    # Find the best fit ideal functions
    chosen_functions = find_best_fit_functions(train_data.values, ideal_data.values)

    # Assign the corresponding ideal function to each dataset in test_data
    for i, chosen_function_index in enumerate(chosen_functions):
        x_test = test_data['x']
        coefficients = np.polyfit(train_data['x'], train_data.iloc[:, chosen_function_index + 1], deg=1)
        assigned_values = np.polyval(coefficients, x_test)
        test_data[f'y{i+1}'] = assigned_values

    # Visualize the assigned datasets
    plt.figure()
    for i, chosen_function_index in enumerate(chosen_functions):
        plt.plot(test_data['x'], test_data[f'y{i+1}'], label=f'Assigned y{i+1}')
    plt.scatter(test_data['x'], test_data['y'], color='red', label='Actual y')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Assigned Datasets')
    plt.legend()
    plt.show()

    # Save the updated test data to a CSV file
    test_data.to_csv('updated_test_data.csv', index=False)

    print("Updated test data saved to 'updated_test_data.csv'")

# Specify the file paths
train_file = 'train.csv'
ideal_file = 'ideal.csv'
test_file = 'test.csv'
database_file = 'data.db'

# Call the main function
main(train_file, ideal_file, test_file, database_file)

def find_best_fit_functions(train_data, ideal_data):
    # Your code for finding the best fit functions here

    return chosen_functions

def main(train_file, ideal_file, test_file, database_file):
    # Your code for reading and processing the data here

    # Create Bokeh plots
    p1 = figure(title="Training Data", x_axis_label='x', y_axis_label='y')
    p2 = figure(title="Test Data", x_axis_label='x', y_axis_label='y')
    p3 = figure(title="Chosen Ideal Functions", x_axis_label='x', y_axis_label='y')
    p4 = figure(title="Assigned Datasets", x_axis_label='x', y_axis_label='y')

    # Plot training data
    p1.circle(train_data['x'], train_data['y'], legend_label='Training Data', color='blue')

    # Plot test data
    p2.circle(test_data['x'], test_data['y'], legend_label='Test Data', color='green')

    # Plot chosen ideal functions
    colors = ['red', 'orange', 'yellow', 'purple', 'cyan']  # Choose colors for each ideal function
    for i, func_idx in enumerate(chosen_functions):
        x_ideal = ideal_data['x']
        y_ideal = ideal_data[f'y{func_idx}']
        p3.line(x_ideal, y_ideal, legend_label=f'Ideal Function {func_idx}', color=colors[i % len(colors)])

    # Plot assigned datasets
    for i, func_idx in enumerate(chosen_functions):
        x_test = test_data['x']
        y_test_assigned = test_data[f'y_assigned_{func_idx}']
        p4.circle(x_test, y_test_assigned, legend_label=f'Dataset assigned to Function {func_idx}', color=colors[i % len(colors)])

    # Create a grid layout for the plots
    grid = gridplot([[p1, p2], [p3, p4]])

    # Show the grid layout
    show(grid)

# Your code for file paths and calling the main function here

# Call output_notebook to display the plots in the notebook
output_notebook()

# Call the main function
main(train_file, ideal_file, test_file, database_file)
