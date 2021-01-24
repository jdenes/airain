from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def classification_perf(y_true, y_pred):
    """
    Computes and formats performance metrics of classifier.

    :param y_true: true labels.
    :param y_pred: predicted labels.
    :return: string to print.
    :rtype: str
    """

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    string = f"- accuracy: {accuracy:.3f}\n- f1 macro: {f1_macro:.3f}\n- f1 micro: {f1_micro:.3f}\n"
    string += f"- recall: {recall:.3f}\n- precision: {precision:.3f}"
    return string


def benchmark_metrics(initial_gamble, balance_hist, orders_hist):
    """
    Computes financial metrics for a single asset.

    :param int initial_gamble: initial amount invested in each asset.
    :param list[int|float] balance_hist: list of asset's balance history.
    :param list[int] orders_hist: list of binary orders at each time step.
    :return: dictionary with various metrics.
    :rtype: dict
    """

    # Final profit
    profit = round(balance_hist[-1] - initial_gamble, 2)

    # Profits and returns
    full_bal_hist = [initial_gamble] + balance_hist
    profits_hist = [round(full_bal_hist[i] - full_bal_hist[i-1], 2) for i in range(1, len(full_bal_hist))]
    returns_hist = [round(100 * profit / initial_gamble, 2) for profit in profits_hist]

    # Basic counts: nb of sell and buy days, and correctness
    buy_count, sell_count, correct_buy, correct_sell = 0, 0, 0, 0
    for i in range(len(balance_hist)):
        is_positive = int(profits_hist[i] >= 0)
        if orders_hist[i] == 1:
            buy_count += 1
            correct_buy += is_positive
        elif orders_hist[i] == 0:
            sell_count += 1
            correct_sell += is_positive
    buy_acc = round(100 * correct_buy / buy_count, 2) if buy_count != 0 else 'NA'
    sell_acc = round(100 * correct_sell / sell_count, 2) if sell_count != 0 else 'NA'

    # Overall daily scores
    pos_pls, neg_pls = [i for i in profits_hist if i >= 0], [i for i in profits_hist if i < 0]
    pos_days = round(100 * sum([i >= 0 for i in profits_hist]) / len(profits_hist), 2)
    mean_returns = round(sum(returns_hist) / len(returns_hist), 2)
    mean_win = round(sum(pos_pls) / len(pos_pls), 2) if len(pos_pls) != 0 else 'NA'
    mean_loss = round(sum(neg_pls) / len(neg_pls), 2) if len(neg_pls) != 0 else 'NA'
    max_win, max_loss = max(profits_hist), min(profits_hist)

    return {'profit': profit,
            'profits_history': profits_hist,
            'positive_days': pos_days,
            'mean_returns': mean_returns,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_accuracy': buy_acc,
            'sell_accuracy': sell_acc,
            'mean_wins': mean_win,
            'mean_loss': mean_loss,
            'max_loss': max_loss,
            'max_win': max_win
            }


def benchmark_portfolio_metric(initial_gamble, assets_balance_hist):
    """
    Computes financial metrics for a portfolio.

    :param int initial_gamble: initial amount invested in each asset.
    :param list[list[int|float]] assets_balance_hist: list of each asset's balance history.
    :return: dictionary with various metrics.
    :rtype: dict
    """

    assets_profits_hist, assets_returns = [], []
    for balance_hist in assets_balance_hist:
        full_bal_hist = [initial_gamble] + balance_hist
        profits_hist = [round(full_bal_hist[i] - full_bal_hist[i - 1], 2) for i in range(1, len(full_bal_hist))]
        returns_hist = [round(100 * profit / initial_gamble, 2) for profit in profits_hist]
        mean_returns = round(sum(returns_hist) / len(returns_hist), 2)
        assets_profits_hist.append(profits_hist), assets_returns.append(mean_returns)
    portfolio_pls_hist = [sum([pls[i] for pls in assets_profits_hist]) for i in range(len(assets_profits_hist[0]))]
    portfolio_mean_profits = round(sum(portfolio_pls_hist) / len(portfolio_pls_hist), 2)
    portfolio_positive_days = round(100 * len([x for x in portfolio_pls_hist if x > 0]) / len(portfolio_pls_hist), 2)
    assets_profits = [balance[-1] - initial_gamble for balance in assets_balance_hist]
    count_profitable_assets = len([x for x in assets_profits if x > 0])
    assets_mean_profits = round(sum(assets_profits) / len(assets_profits), 2)
    assets_mean_returns = round(sum(assets_returns) / len(assets_returns), 3)

    return {'assets_returns': assets_returns,
            'assets_mean_profits': assets_mean_profits,
            'assets_mean_returns': assets_mean_returns,
            'portfolio_mean_profits': portfolio_mean_profits,
            'portfolio_positive_days': portfolio_positive_days,
            'count_profitable_assets': count_profitable_assets,
            }
