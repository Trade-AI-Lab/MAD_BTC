import os
import optuna
import plotly.graph_objs as go
import plotly.io as pio
import sys
from matplotlib.colors import to_rgb
import numpy as np
from matplotlib.colors import to_hex
# Correct path to your project directory
absolute_path = os.path.abspath(__file__)[
    : os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")
]
sys.path.append(absolute_path)
from src.utils.utils_df import open_dataframe

if __name__ == "__main__":
    list_studies = os.listdir(absolute_path + "/study")
    TIME = "1d"
    os.makedirs("robustness_plot", exist_ok=True)
    list_crypto_currency = []
    list_params_x_mean = []
    list_params_y_mean = []
    list_std_dev_open = []
    
    for study_name in list_studies:
        study_name = study_name.split(".")[0]
        PATH = os.path.join(absolute_path, "study/", study_name)
        storage_name = f"sqlite:///{PATH}.db"
        study = optuna.study.load_study(study_name=study_name, storage=storage_name)
        list_study_name_split = study_name.split("_")
        crypto_currency = list_study_name_split[3]
        exchange = list_study_name_split[4]
        time = list_study_name_split[5]
        
        if time == TIME:
            df = study.trials_dataframe()
            df_sorted = df.sort_values(by='value', ascending=False)
            top_df = df_sorted.head(25)
            std_dev_open = open_dataframe(r"data/"+f"{exchange}-{crypto_currency}-{time}.pkl")["open"].pct_change().std()
            list_crypto_currency += 25 * [crypto_currency]
            list_params_x_mean += list(top_df["params_x"].values)
            list_params_y_mean += list(top_df["params_y"].values)
            list_std_dev_open += 25 * [std_dev_open]
    
    def interpolate_colors(start_color, end_color, num_colors):
        """ Generate a gradient of colors between start_color and end_color """
        start_rgb = np.array(to_rgb(start_color))
        end_rgb = np.array(to_rgb(end_color))
        
        gradient = np.linspace(start_rgb, end_rgb, num_colors)
        return [to_hex(tuple(rgb)) for rgb in gradient]

    # Create a gradient from blue to red
    sort_list = sorted(list_crypto_currency)
    colors = interpolate_colors('blue', 'red', len(sort_list))

    # Create a color mapping dictionary
    name_to_color = dict(zip(sort_list, colors))

    # Map the names to their respective colors
    marker_colors = [name_to_color[crypto] for crypto in list_crypto_currency]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=list_params_x_mean,
        y=list_params_y_mean,
        z=list_std_dev_open,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8,
            color=marker_colors
        ),
        text=list_crypto_currency,  # Assign names to the text attribute
        hoverinfo='text'  # Display the text when hovering
    )])

    # Add title and axis labels
    fig.update_layout(
        title='3D Scatter Plot',
        scene=dict(
            xaxis_title='list_params_x_mean',
            yaxis_title='list_params_y_mean',
            zaxis_title='list_std_dev_open'
        )
    )

    # Show the plot
    pio.show(fig)
