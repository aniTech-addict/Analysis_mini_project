# Advanced Data Analysis in Python

This project demonstrates advanced data analysis techniques using Python libraries: NumPy, pandas, and Matplotlib.

The script `main.py` performs the following operations:
- Creates and manipulates sample sales data using NumPy.
- Performs data manipulation, aggregation, and calculates growth rates using pandas DataFrames.
- Generates various visualizations (line plot, bar plot, histogram, boxplot, heatmap) using Matplotlib to illustrate sales trends and distributions.
- Includes basic advanced analysis like calculating a rolling average and simple forecasting using linear regression.

## Libraries Used

- **NumPy**: For numerical operations and array manipulation.
- **pandas**: For data manipulation and analysis.
- **Matplotlib**: For creating static, interactive, and animated visualizations.

## How to Run

1.  Ensure you have Python installed.
2.  Install the required libraries:
    ```bash
    pip install numpy pandas matplotlib
    ```
3.  Run the script:
    ```bash
    python main.py
    ```

Note: The script will print analysis results to the console. The Matplotlib plots are generated but commented out with `# plt.show()` in the script, as they require a graphical environment to display. To view the plots, uncomment the `plt.show()` lines in `main.py`.
