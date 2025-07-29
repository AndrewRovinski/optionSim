#!/usr/bin/env python3
"""
PLOTTING ROUTINES TO ILLUSTRATE SCENARIOS

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as colors

from optionsim.volatilitysurface import VolatilitySurface

sns.set(rc={'figure.figsize': (12, 9)})


# Function to format Y axis on charts

def dollars(x, pos):
    return '$%1.0f' % x


def thousands(x, pos):
    return '$%1.1fK' % (x*1e-3)


def millions(x, pos):
    return '$%1.1fM' % (x*1e-6)


class MidpointNormalize(colors.Normalize):
    """
    Class which normalizes color palettes to diverge from a set midpoint
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        """
        :param vmin: minimum value to use
        :param vmax: maximum value to use
        :param midpoint: midpoint to normalize around
        :param clip: boolean for clipping
        """
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """
        :param value: value to normalize
        :param clip: boolean for clipping
        :return: normalized value
        """

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def scenario_percentile_rank(log_name, variable, percentile_rank):
    """
    Function that takes a logfile, a variable, and a percentile rank,
    then returns the scenario data which matches the input criteria.

    :param log_name:
    :param variable:
    :param percentile_rank:
    :return:
    """
    data = pd.read_csv(log_name)
    final = data[data['time_step'] == data['time_step'].max()]
    pct_rank = final[variable].rank(method='max', pct=True)
    closeness = abs(pct_rank - percentile_rank)
    closest = closeness.idxmin()
    scenario = final.loc[closest, 'Scenario Number']
    out = data[data['Scenario Number'] == scenario]
    return out


def plot_equity(data: pd.DataFrame, ax: plt.axis,
                index_name: str, label_name: str) -> plt.axis:

    sns.lineplot(data['time_step'], data[index_name], ax=ax, label=label_name)
    ax.set(ylabel='Index Level')
    return ax


def plot_scenario_pnl(data: pd.DataFrame, ax: plt.axis,
                      detailed: bool = True) -> plt.axis:
    """

    :param data:
    :param ax:
    :param detailed:
    :return:
    """

    if detailed:
        sns.lineplot(data['time_step'], data['Total Assets'],
                     label='Total Assets', color='c', ax=ax)

        sns.lineplot(data['time_step'], data['Liability Value'],
                     label='Liability Value', color='m', ax=ax)

    sns.lineplot(data['time_step'], data['PnL'], color='k', label='P&L', ax=ax)
    ax.fill_between(data['time_step'], data['PnL'], where=data['PnL'] > 0,
                    facecolor='g', alpha=0.3)

    ax.fill_between(data['time_step'], data['PnL'], where=data['PnL'] < 0,
                    facecolor='r', alpha=0.3)

    ax.set_xlabel('Time Step')
    ax.set_ylabel('P&L')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(millions))
    return ax


def plot_cash_flows(data: pd.DataFrame, ax: plt.axis) -> plt.axis:
    """

    :param data:
    :param ax:
    :return:
    """

    cash_flows = ['Cash Flow from Expirations', 'Cash Flow from Hedge Spend',
                  'Cash Flow from Mark to Market', 'Interest Earned']

    cumulative = data[cash_flows].cumsum()
    cumulative.plot(ax=ax)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(millions))
    return ax


def plot_scenario(log_filepath, fig_title, out_filepath,
                  scenario_number):
    """
    Plots a single scenario data to show the asset/liabilities values, as
    well as the overall P&L at each time step.

    :param log_filepath: string of the log filepath
    :param fig_title: string of the figure title
    :param out_filepath: string of the filepath to save the output
    :param scenario_number: int of the scenario number to plot
    :return: plot of the data
    """

    data = pd.read_csv(f'./logs/{log_filepath}.csv')
    data = data[data['Scenario Number'] == scenario_number]
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all')
    fig.suptitle(fig_title)

    # plot price chart
    plot_equity(data, axes[0], 'spx_level', 'S&P 500 Price Index')

    # plot P&L
    plot_scenario_pnl(data, axes[1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'./analysis/scenarios/{out_filepath}.png')


def plot_scenario_cashflows(log_filepath, scenario_number, name, out_filename):
    """

    :param log_filepath:
    :param scenario_number:
    :param name:
    :param out_filename:
    :return:
    """

    data = pd.read_csv(f'./logs/{log_filepath}.csv')
    data = data[data['Scenario Number'] == scenario_number]

    # Calculate cumulative totals
    cash_flows = ['Cash Flow from Expirations', 'Cash Flow from Hedge Spend',
                  'Cash Flow from Mark to Market', 'Interest Earned']

    # cumulative = data[cash_flows].groupby(data.index // 13).sum().cumsum()
    cumulative = data[cash_flows].cumsum()

    fig, axes = plt.subplots()

    cumulative.plot(ax=axes)

    formatter = mtick.FuncFormatter(millions)
    axes.yaxis.set_major_formatter(formatter)
    fig.suptitle(name)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'./analysis/scenarios/{out_filename}.png')
    plt.show()


def plot_gbm_equity(log_filepath, out_filepath):
    """

    :param log_filepath:
    :param out_filepath:
    :return:
    """

    data = pd.read_csv(f'./scenarioCSV/{log_filepath}.csv')
    fig, axes = plt.subplots()
    fig.suptitle('GBM Equity Paths')
    axes.plot(data)
    axes.set(title='S&P500 Price Index', ylabel='Index Level')
    fig.savefig(out_filepath)


def plot_gbm_pnl(log_filepath, strategy_name, out_filename,
                 yaxis_limits: list = None):

    data = pd.read_csv(f'./logs/{log_filepath}.csv')
    results = data[data['time_step'] == data['time_step'].max()]
    results = results.sort_values(by='PnL')
    results = results.reset_index()
    results.index = results.index + 1
    results.index.rename('Scenario Rank', inplace=True)

    fig, axes = plt.subplots()

    # plot P&L
    axes.plot(results['PnL'], color='k', label='PnL')
    axes.fill_between(results.index, results['PnL'],
                      where=results['PnL'] > 0, facecolor='g', alpha=0.3)

    axes.fill_between(results.index, results['PnL'],
                      where=results['PnL'] < 0, facecolor='r', alpha=0.3)

    if yaxis_limits:
        axes.set_ylim(bottom=yaxis_limits[0], top=yaxis_limits[1])

    axes.set_xlabel('Sorted Scenario Rank')
    axes.set_ylabel('P&L ($)')
    formatter = mtick.FuncFormatter(thousands)
    axes.yaxis.set_major_formatter(formatter)
    plt.title('{} Strategy P&L'.format(strategy_name))
    plt.tight_layout()
    fig.savefig(f'./analysis/P&L dist/{out_filename}.png')


def plot_gbm_pnl_comparison(log_config, out_filename):
    """

    :param log_config:
    :param out_filename:
    :return:
    """

    fig, axes = plt.subplots()

    for strategy_name, log_file in log_config.items():
        df = pd.read_csv(f'./logs/{log_file}.csv')
        results = df[df['time_step'] == df['time_step'].max()]
        results = results[['Scenario Number', 'PnL']]
        results = results.sort_values(by='PnL')
        results = results.reset_index()
        results.index = results.index + 1
        results.index.rename('Scenario Rank', inplace=True)
        axes.plot(results['PnL'], label=strategy_name)

    axes.set_xlabel('Sorted Scenario Rank')
    axes.set_ylabel('P&L ($)')

    formatter = mtick.FuncFormatter(thousands)
    axes.yaxis.set_major_formatter(formatter)
    plt.legend()
    plt.tight_layout()

    fig.savefig(f'./analysis/P&L dist/{out_filename}.png')


def plot_scenario_spx_pnl_joint(log_filepath, strategy_name, out_filename,
                                yaxis_limits=None):
    """

    :param log_filepath:
    :param strategy_name:
    :param out_filename:
    :return:
    """

    fig, axes = plt.subplots()

    data = pd.read_csv(f'./logs/{log_filepath}.csv')

    results = data[data['time_step'] == data['time_step'].max()]
    vmin = np.min(results['PnL']) * 0.5
    vmax = np.max(results['PnL']) * 0.5
    norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

    axes = sns.scatterplot('spx_level', 'PnL', data=results, hue='PnL',
                           palette='Spectral', hue_norm=norm, edgecolor='k')

    if yaxis_limits:
        axes.set_ylim(bottom=yaxis_limits[0], top=yaxis_limits[1])

    axes.set_xlabel('Terminal SPX Level')
    axes.set_ylabel('P&L')

    axes.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    formatter = mtick.FuncFormatter(millions)
    axes.yaxis.set_major_formatter(formatter)
    leg = plt.legend()

    leg_title = leg.get_texts()[0]
    leg_title.set_text('P&L')

    for label in leg.get_texts()[1:]:
        f_label = float(label._text.strip('"')) / 10 ** 6
        label.set_text('$%1.0fM' % f_label)

    plt.title('{} P&L by Terminal Index Level'.format(
        strategy_name))

    plt.tight_layout()
    fig.savefig(f'./analysis/SPX x PnL/{out_filename}.png')


def plot_vol_surface(surface_name: str, plot_title: str,
                     out_filename: str) -> None:
    """
    Generates a 3d plot of the volatility surface from the CSV file

    :param surface_name: string of the file in the 'scenarioCSV' directory
    :param plot_title: string of the title for the plot
    :param out_filename: string of the plot file to save in 'analysis' folder
    :return: None
    """

    vols = pd.read_csv(f'./scenarioCSV/{surface_name}.csv', header=None)

    # Create the volatility surface object
    expirations = np.array(vols.iloc[0].dropna())
    strikes = np.array(vols.iloc[:, 0].dropna())
    values = np.array(vols.iloc[1:, 1:])
    surface = VolatilitySurface(expirations, strikes, values)

    # Create a fine grid for interpolation
    x = np.linspace(0.25, expirations[-1], 53)
    y = np.linspace(strikes[0], strikes[-1], 100)
    xx, yy = np.meshgrid(x, y)

    # Get the results
    zz = surface.iv_grid(xx.ravel(), yy.ravel()).reshape(xx.shape)

    result = pd.DataFrame(zz, index=y, columns=x)
    result.to_csv('./scenarioCSV/volsurface_interp.csv')

    fig = plt.figure(figsize=[11, 8])
    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                    cmap='plasma', edgecolor='none')

    ax.set(xlabel='Years to Maturity', ylabel='Percent Strike', zlabel='IV')
    plt.title(plot_title)
    plt.tight_layout()
    fig.savefig(f'./analysis/{out_filename}.png')
    plt.show()


def scenario_statistics(strategies, scenario_numbers,
                        terminal_vars, cumulative_vars):

    output = []
    for strat_name, log_path in strategies.items():

        data = pd.read_csv(f'./logs/{log_path}.csv')
        data = data[data['Scenario Number'].isin(scenario_numbers)]

        terminal = data.loc[
            data.groupby('Scenario Number')['time_step'].idxmax()]

        terminal = terminal.set_index('Scenario Number', drop=True)
        terminal = terminal[terminal_vars]

        cumulative = data.groupby('Scenario Number')[cumulative_vars].agg(sum)
        stats = pd.merge(terminal, cumulative, left_index=True,
                         right_index=True)

        stats['strategy'] = strat_name
        output.append(stats)

    output = pd.concat(output)
    output.to_csv('./analysis/Scenario Statistics.csv')


def run_analysis(logfile, scenario_numbers, vols, name, out, strategies,
                 percentiles, terminal_vars, cumulative_vars,
                 analysis_type, yaxis_limits=None):
    """

    :param logfile:
    :param scenario_numbers:
    :param vols:
    :param name:
    :param out:
    :param strategies:
    :param percentiles:
    :param analysis_type:
    :return:
    """

    if analysis_type == 'scenario':
        for scenario_num in scenario_numbers:
            plot_scenario(logfile, name, f'{out} {scenario_num}', scenario_num)

    elif analysis_type == 'vols':
        plot_vol_surface(vols, name, out)

    elif analysis_type == 'gbm_pnl':
        plot_gbm_pnl(logfile, name, out, yaxis_limits=yaxis_limits)

    elif analysis_type == 'gbm_equity':
        plot_gbm_equity(logfile, out)

    elif analysis_type == 'gbm_comparison':
        plot_gbm_pnl_comparison(strategies, out)

    elif analysis_type == 'gbm_percentiles':

        for pct in percentiles:

            rank = scenario_percentile_rank(
                f'./logs/{logfile}.csv', 'PnL', pct)

            pctile = int(pct * 100)

            rank.to_csv(f'./logs/single scenarios/'
                        f'{logfile}_{pctile}percentile.csv',
                        index=False)

            plot_scenario(f'/single scenarios/{logfile}_{pctile}percentile',
                          f'{name}',
                          f'{out} {pctile} percentile')

    elif analysis_type == 'spx_pnl_joint':
        plot_scenario_spx_pnl_joint(logfile, name, out, yaxis_limits)

    elif analysis_type == 'cashflow':
        plot_scenario_cashflows(logfile, scenario_numbers, name, out)

    elif analysis_type == 'scenario stats':
        scenario_statistics(strategies, scenario_numbers,
                            terminal_vars, cumulative_vars)

    else:
        raise NotImplementedError


if __name__ == '__main__':

    # Plot type to output
    analysis = 'gbm_comparison'

    # Name of the log to use from the 'logs' directory
    log = 'B&H 3M 3% Risk Reversal 7%drift 17.75%RV medianIV'

    # Strategy names and log files to use for comparisons
    strategies = {

        'B&H 3% OTM Risk Reversal':
            'B&H 3M 3% Risk Reversal 7%drift 17.75%RV medianIV',
        #'B&H 5% OTM Risk Reversal':
        #    'B&H 5% OTM Risk Reversal 7%drift 17.75%RV medianIV',
        #'B&H 10% OTM Risk Reversal':
        #    'B&H 10% OTM Risk Reversal 7%drift 17.75%RV medianIV'

    }

    # Name for the plot title
    name = 'Risk Reversal OTM Comparison 7%drift 17.75%RV medianIV'

    scenario_numbers = [1] #[43, 788, 924, 177, 797, 648, 891]

    # Volatility surface to use
    vols = 'volsurface_hist'

    yaxis_limits = None
    percentiles = [0.05, 0.5, 0.95]

    terminal_vars = ['Total Assets',
                     'PnL']

    cumulative_vars = ['Interest Earned',
                       'Cash Flow from Hedge Spend',
                       'Cash Flow from Expirations']

    # Run the analysis
    run_analysis(
        logfile=log,
        scenario_numbers=scenario_numbers,
        vols=vols,
        name=name,
        out=name,
        strategies=strategies,
        percentiles=percentiles,
        terminal_vars=terminal_vars,
        cumulative_vars=cumulative_vars,
        analysis_type=analysis,
        yaxis_limits=yaxis_limits)


