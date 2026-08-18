"""Reusable plotting helpers.

Pulls the dual-axis dispatch plot, the prediction-vs-actual line plot, and
the profit-comparison bar chart out of the ~6 near-identical copies that were
scattered through the original notebook.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_price_and_dispatch(df, title, n_steps=168, forecasted_price=None, price_label="Price (€/MWh)"):
    """Dual-axis plot: price (left axis) plus charge/discharge power (right axis).

    ``df`` needs columns 'price', 'P_ch', 'P_dis' and a datetime index.
    If ``forecasted_price`` (a Series aligned to df's index) is given, it is
    overlaid faintly for comparison against the actual price.
    """
    subset = df.head(n_steps)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if forecasted_price is not None:
        ax1.plot(subset.index, forecasted_price.head(n_steps), color="blue", alpha=0.3,
                 label="Forecasted Price (€/MWh)")
    ax1.plot(subset.index, subset["price"], color="blue", label=price_label)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel(price_label, color="blue", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="blue", labelsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(subset.index, subset["P_ch"], color="red", label="Charge Power (kW)")
    ax2.plot(subset.index, subset["P_dis"], color="green", label="Discharge Power (kW)")
    ax2.set_ylabel("Power (kW)", color="black", fontsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    plt.title(title, fontsize=14)
    fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.25), fontsize=12)
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_prediction_vs_actual(actual, predicted, title, xlabel="Time (Hours)", n_steps=None, ylim=(-70, 250)):
    """Line plot comparing actual vs. predicted Day-Ahead prices."""
    if n_steps is not None:
        actual = actual[:n_steps]
        predicted = predicted[:n_steps]
    time_index = range(len(actual))

    plt.figure(figsize=(14, 6))
    plt.plot(time_index, actual, label="Actual", color="black", linewidth=2)
    plt.plot(time_index, predicted, label="Random Forest Prediction", linestyle="--")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Day-Ahead price (€/MWh)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(*ylim)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    plt.show()


def plot_sensitivity_correction(actual, forecasted, corrected_positive, corrected_negative, title, n_steps=169):
    """Overlay actual/forecast/±-corrected prices for a sensitivity-analysis check."""
    time_index = range(n_steps)

    plt.figure(figsize=(14, 6))
    plt.plot(time_index, actual[:n_steps], label="Actual", color="black", linewidth=4)
    plt.plot(time_index, forecasted[:n_steps], label="Random Forest Prediction", linestyle="-")
    plt.plot(time_index, corrected_positive[:n_steps], label="Corrected Positive", linestyle="--")
    plt.plot(time_index, corrected_negative[:n_steps], label="Corrected Negative", linestyle="--")

    plt.xlabel("Time (hours)", fontsize=12)
    plt.ylabel("Day-Ahead Price (€/MWh)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.ylim(-70, 250)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances, title="Random Forest: Top Feature Importances of Day Ahead Price"):
    fig, ax = plt.subplots(figsize=(10, 6))
    importances.plot(kind="barh", ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_profit_comparison(labels, profit_percentages, colors, title="Comparison of Profits Across Different Scenarios"):
    """Bar chart of profit as % of theoretical maximum, with shaded loss on top of each bar.

    ``labels[0]``/``profit_percentages[0]`` is expected to be the 100% theoretical baseline.
    """
    loss_percentages = [100 - p for p in profit_percentages]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, profit_percentages, color=colors)

    for label, profit_pct, loss_pct, color in zip(labels[1:], profit_percentages[1:], loss_percentages[1:], colors[1:]):
        plt.bar(label, loss_pct, bottom=profit_pct, color=color, alpha=0.2, hatch="//", ec="white")

    plt.ylabel("Profit (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(80, 105)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, 85, f"{yval:.2f}%", ha="center", va="center", fontsize=10)

    for label, profit_pct, loss_pct, color in zip(labels[1:], profit_percentages[1:], loss_percentages[1:], colors[1:]):
        bar = plt.bar(label, loss_pct, bottom=profit_pct, color=color, alpha=0.5, hatch="//", ec="white")[0]
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + yval / 2, f"{yval:.2f}% Loss",
                  ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_profit_delta_vs_error(fractions_percent, percentage_differences, xlabel, title="Percentage Difference in Profit vs. Percentage Change in Error"):
    plt.figure(figsize=(10, 6))
    plt.plot(fractions_percent, percentage_differences, marker="o", linestyle="-")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Percentage Difference in Profit (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.tight_layout()
    plt.show()
