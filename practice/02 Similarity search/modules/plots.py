import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# for visualization
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)


def plot_ts_set(ts_set: np.ndarray, title: str = 'Input Time Series Set') -> None:
    """
    Plot the time series set

    Parameters
    ----------
    ts_set: time series set
    title: title of plot
    """

    ts_num, m = ts_set.shape

    fig = go.Figure()

    for i in range(ts_num):
        fig.add_trace(go.Scatter(x=np.arange(m), y=ts_set[i], line=dict(width=3), name="Time series " + str(i)))

    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title='Values',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)
    fig.update_layout(title=title,
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="colab")


def mplot2d(x: np.ndarray, y: np.ndarray, plot_title: str = None, x_title: str = None, y_title: str = None, trace_titles: np.ndarray = None) -> None:
    """
    Multiple 2D Plots on figure for different experiments

    Parameters
    ----------
    x: values of x axis of plot
    y: values of y axis of plot
    plot_title: title of plot
    x_title: title of x axis of plot
    y_title: title of y axis of plot
    trace_titles: titles of plot traces (lines)
    """

    fig = go.Figure()

    for i in range(y.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=y[i], line=dict(width=3), name=trace_titles[i]))

    fig.update_xaxes(showgrid=False,
                     title=x_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2,
                     tickvals=x)
    fig.update_yaxes(showgrid=False,
                     title=y_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black')),
                      width=1000,
                      height=600
                      )

    fig.show(renderer="colab")


def plot_bestmatch_data(ts: np.ndarray, query: np.ndarray) -> None:
    """
    Visualize the input data (time series and query) for the best match task

    Parameters
    ----------
    ts: time series
    query: query
    """

    query_len = query.shape[0]
    ts_len = ts.shape[0]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.1, 0.9], subplot_titles=("Query", "Time Series"), horizontal_spacing=0.04)

    fig.add_trace(go.Scatter(x=np.arange(query_len), y=query, line=dict(color=px.colors.qualitative.Plotly[1])),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(ts_len), y=ts, line=dict(color=px.colors.qualitative.Plotly[0])),
                row=1, col=2)

    fig.update_annotations(font=dict(size=24, color='black'))

    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      title_x=0.5)

    fig.show(renderer="colab")


def plot_bestmatch_results(ts: np.ndarray, query: np.ndarray, bestmatch_results: dict) -> None:
    """
    Visualize the best match results

    Parameters
    ----------
    ts: time series
    query: query
    bestmatch_results: output data found by the best match algorithm
    """

    # INSERT YOUR CODE
    # --- Попытка извлечь индекс лучшего совпадения ---
    best_idx = (
        bestmatch_results.get("index")
        or bestmatch_results.get("best_index")
        or (bestmatch_results.get("indices")[0] if isinstance(bestmatch_results.get("indices"), (list, np.ndarray)) else None)
    )

    # --- Попытка извлечь расстояние ---
    distance = (
        bestmatch_results.get("distance")
        or bestmatch_results.get("best_distance")
        or (bestmatch_results.get("distances")[0] if isinstance(bestmatch_results.get("distances"), (list, np.ndarray)) else None)
    )

    # Проверка наличия индекса
    if best_idx is None:
        raise ValueError(
            "Не удалось определить индекс совпадения. "
            "Убедитесь, что в bestmatch_results есть ключ 'index', 'best_index' или 'indices'."
        )

    # Извлекаем участок временного ряда, соответствующий лучшему совпадению
    match_segment = ts[best_idx:best_idx + len(query)]

    plt.figure(figsize=(12, 6))

    # Весь временной ряд
    plt.plot(ts, label="Временной ряд", color="gray", alpha=0.6)

    # Лучшее совпадение
    plt.plot(range(best_idx, best_idx + len(query)), match_segment,
             label=f"Лучшее совпадение (начало = {best_idx})", color="orange", linewidth=2)

    # Запрос (масштабированный для наглядного сравнения)
    plt.plot(range(best_idx, best_idx + len(query)),
             (query - np.mean(query)) / np.std(query) * np.std(match_segment) + np.mean(match_segment),
             label="Запрос (масштабированный)", color="blue", linestyle="--")

    # Заголовок
    if distance is not None:
        plt.title(f"Лучшее совпадение (расстояние = {distance:.4f})")
    else:
        plt.title("Лучшее совпадение")

    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def pie_chart(labels: np.ndarray, values: np.ndarray, plot_title='Pie chart') -> None:
    """
    Build the pie chart

    Parameters
    ----------
    labels: sector labels
    values: values
    """

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_traces(textfont_size=20)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      legend=dict(font=dict(size=20, color='black')),
                      width=700,
                      height=500
                      )

    fig.show(renderer="colab")
