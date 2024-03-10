import os
import optuna
import plotly.express as px
absolute_path = os.path.abspath(__file__)[:os.path.abspath(__file__).find("MAD_BTC")+len("MAD_BTC")]
if __name__=='__main__':
    study_name = "study_momentum_strategy_ADAUSDT_binance_4h"
    PATH = os.path.join(absolute_path ,"study/",study_name)
    storage_name = f"sqlite:///{PATH}.db"
    study = optuna.study.load_study(study_name=study_name, storage=storage_name)
    df = study.trials_dataframe()
    fig = px.scatter_3d(
            df,
            x='value',
            y='params_x',
            z='params_y',
            color='params_ratio_method',  # You can use another metric for coloring points
            title='Optimized Parameters for the ' + study_name.replace('_', ' '),
            labels={'params_ratio_method': 'Ratio Method', 'params_x': 'X', 'params_y': 'Y'},
        )
    fig.show()
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_new = df[['value', 'params_x', 'params_y']].dropna()
    scaled_data = scaler.fit_transform(df_new)

    
    # Choose the number of clusters (k)
    k = 6
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k)
    df_new['cluster'] = kmeans.fit_predict(scaled_data)
    fig = px.scatter_3d(
            df_new,
            x='value',
            y='params_x',
            z='params_y',
            color='cluster',  # You can use another metric for coloring points
            title='Optimized Parameters for the ' + study_name.replace('_', ' '),
            labels={'params_ratio_method': 'Ratio Method', 'params_x': 'X', 'params_y': 'Y'},
        )
    fig.show()

