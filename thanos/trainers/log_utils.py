import numpy as np
import wandb
import plotly.graph_objs as go


def get_wandb_confusion_matrix_plot(confusion_matrix, label_names=None):
    """
    Given a confusion matrix, return a wandb compatible plot object to be logged

    Parameters
    ----------
    confusion_matrix: 2D matrix
        targets = lines, predictions = columns

    Returns
    -------
    wandb.data_types.Plotly
    """
    if label_names is None:
        label_names = list(range(confusion_matrix.shape[0]))
    confmatrix = go.Heatmap({
        'x': label_names, 'y': label_names, 'z': confusion_matrix,
        "colorscale": "Greens"
    })
    transparent = 'rgba(0, 0, 0, 0)'
    fig = go.Figure((confmatrix, ))
    xaxis = {'title':{'text':'Predictions'}, 'showticklabels':True}
    yaxis = {'title':{'text':'Targets'}, 'showticklabels':True}
    fig.update_layout(
        title={'text':'Confusion matrix', 'x':0.5}, 
        paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)

    return wandb.data_types.Plotly(fig)
