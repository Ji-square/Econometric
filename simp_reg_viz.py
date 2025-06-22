import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
def plot_regression(x, y, data, results, title, add_ci=True):
    """Create an enhanced regression plot with confidence intervals and annotations.

    Parameters
    ----------
    x : str
        Name of x variable in data
    y : str
        Name of y variable in data
    data : pandas.DataFrame
        Data containing x and y
    results : statsmodels results object
        Fitted regression results
    title : str
        Plot title
    add_ci : bool
        Whether to add confidence intervals

    """
    fig, ax = plt.subplots(
        figsize=(10, 6),
    )  # Create a figure and an axes object for the plot

    # Scatter plot of the actual data points
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        alpha=0.5,
        ax=ax,
    )  # Scatter plot of x vs y with some transparency

    # Regression line - Calculate predicted y values for a range of x values
    x_range = np.linspace(
        data[x].min(),
        data[x].max(),
        100,
    )  # Generate 100 evenly spaced x values
    y_pred = (
        results.params[0] + results.params[1] * x_range
    )  # Calculate predicted y values using the regression equation

    plt.plot(
        x_range,
        y_pred,
        color="red",
        label="Regression Line",
    )  # Plot the regression line in red

    if add_ci:
        # Add confidence intervals - Calculate confidence intervals for the predicted values
        y_hat = results.get_prediction(
            pd.DataFrame({x: x_range}),
        )  # Get prediction object for the x_range
        ci = y_hat.conf_int()  # Extract confidence interval bounds
        plt.fill_between(  # Fill the area between the confidence interval bounds
            x_range,
            ci[:, 0],  # Lower bound of confidence interval
            ci[:, 1],  # Upper bound of confidence interval
            color="red",
            alpha=0.1,  # Set transparency of the shaded area
            label="95% CI",  # Label for the confidence interval
        )

    # Add equation and R-squared as text annotations on the plot
    eq = f"y = {results.params[0]:.2f} + {results.params[1]:.2f}x"  # Format the regression equation
    r2 = f"R² = {results.rsquared:.3f}"  # Format the R-squared value
    plt.text(  # Add text to the plot
        0.05,
        0.95,
        eq + "\n" + r2,
        transform=ax.transAxes,  # Specify that coordinates are relative to the axes
        verticalalignment="top",  # Align text to the top
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),  # Add a white box around the text for better readability
    )

    plt.title(title)  # Set the title of the plot
    plt.xlabel(x)  # Set the label for the x-axis
    plt.ylabel(y)  # Set the label for the y-axis
    plt.legend()  # Display the legend
    plt.grid(True, alpha=0.3)  # Add grid lines with transparency
    plt.tight_layout()  # Adjust plot layout to prevent labels from overlapping
