import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


COLORS = ['#f77189', '#e68332', '#bb9832', '#97a431', '#50b131', '#34af84', '#36ada4', '#38aabf', '#3ba3ec',
          '#a48cf4', '#e866f4', '#f668c2']


def nice_plot(ind, curves_list, names_list, title):
    """
    Provides nice plot for profit backtest curves.

    :param list[str] ind: list of date for each backtest step, in format '%Y-%m-%d'.
    :param list[list[float]] curves_list: a list of lists to plot.
    :param list[str] names_list: list of names for each curve.
    :param str title: name of the plot.
    :rtype: None
    """

    font_manager._rebuild()
    plt.rcParams['font.family'] = 'Lato'
    plt.rcParams['font.sans-serif'] = 'Lato'
    plt.rcParams['font.weight'] = 500
    from datetime import datetime
    ind = [datetime.strptime(x, '%Y-%m-%d') for x in ind]
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, x in enumerate(curves_list):
        # c_list = ['gray']
        # for j in range(len(x) - 1):
        #     start, stop = x[j], x[j + 1]
        #     color = ['green', 'red', 'orange'][0 if stop - start > 0 else 1 if stop - start < 0 else 2]
        #     c_list.append(color)
        #     ax.plot([ind[j], ind[j + 1]], [start, stop], linewidth=2, color=color)
        #     # ax.fill_between([ind[j], ind[j+1]], [start, stop], alpha=0.2, color=color)
        # for j in range(len(x)):
        #     ax.plot(ind[j], x[j], '.', color=c_list[j])
        pd.Series(index=ind, data=list(x)).plot(linewidth=2, color=COLORS[i], ax=ax, label=names_list[i], style='.-')
    ax.tick_params(labelsize=12)
    # formatter = dates.DateFormatter('%d/%m/%Y')
    # ax.xaxis.set_major_formatter(formatter)
    ax.spines['right'].set_edgecolor('lightgray')
    ax.spines['top'].set_edgecolor('lightgray')
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.title(title, fontweight=500, fontsize=25, loc='left')
    plt.legend(loc='upper left', fontsize=15)
    plt.grid(alpha=0.3)
    # plt.savefig(path, bbox_inches='tight',format="png", dpi=300, transparent=True)
    plt.show()
