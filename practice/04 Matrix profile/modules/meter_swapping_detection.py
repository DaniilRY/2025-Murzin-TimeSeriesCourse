import numpy as np
import datetime

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)

from modules.mp import *


def heads_tails(consumptions: dict, cutoff, house_idx: list) -> dict, dict:
    """
    Split time series into two parts: Head and Tail

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses

    Returns
    --------
    heads: heads of time series
    tails: tails of time series
    """

    heads, tails = {}, {}
    for i in house_idx:
        heads[f'H_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index < cutoff]
        tails[f'T_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index >= cutoff]
    
    return heads, tails

def compute_new_mp(ts1, m, exclusion_zone=None, ts2=None):
    if exclusion_zone is None:
        exclusion_zone = math.ceil(m / config.STUMPY_EXCL_ZONE_DENOM)

    m = int(m)

    if ts2 is None:
        mp = stumpy.stump(ts1, m)
    else:
        mp = stumpy.stump(ts1, m, ts2)
    
    return mp

def meter_swapping_detection(heads, tails, house_idx, m):
    m = int(m)
    min_score = {'score': float('inf'), 'i': None, 'mp_j': None}

    for i in house_idx:
        head_i = heads[f'H_{i}']
        for j in house_idx:
            if i == j:
                continue
            
            tail_j = tails[f'T_{j}']

            if hasattr(head_i, 'values'):
                head_i_vals = head_i.values.flatten()
            else:
                head_i_vals = head_i.flatten()
            
            if hasattr(tail_j, 'values'):
                tail_j_vals = tail_j.values.flatten()
            else:
                tail_j_vals = tail_j.flatten()
            
            mp = compute_new_mp(head_i_vals, m, ts2=tail_j_vals)
            min_distance = np.min(mp[:, 0])

            if min_distance < min_score['score']:
                min_score['score'] = min_distance
                min_score['i'] = i
                min_score['j'] = j
                min_score['mp_j'] = mp
    
    return min_score


def plot_consumptions_ts(consumptions: dict, cutoff, house_idx: list):
    """
    Plot a set of input time series and cutoff vertical line

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(go.Scatter(x=list(consumptions.values())[i].index, y=list(consumptions.values())[i].iloc[:,0], name=f"House {house_idx[i]}"), row=i+1, col=1)
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red",  row=i+1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)', 
                      height=800,
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="colab")
