import os
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

from analysis import welch_t_test, bootstrap_mean_diff, fit_linear, fit_logistic

DATA_PATH = os.environ.get("DATA_PATH","data/dataset_estudiantes.csv")
df = pd.read_csv(DATA_PATH)

app = Dash(__name__, title="Proyecto Final — Metodos Estadisticos")
server = app.server

def kpi_card(title, value, fmt="{:.3f}"):
    if isinstance(value, (int, float, np.floating)):
        s = fmt.format(value)
    else:
        s = str(value)
    return html.Div(style={
        "padding":"12px","border":"1px solid #eee","borderRadius":"12px",
        "boxShadow":"0 2px 8px rgba(0,0,0,0.05)","background":"#fff"
    }, children=[
        html.Div(title, style={"fontSize":"14px","color":"#666"}),
        html.Div(s, style={"fontSize":"24px","fontWeight":"700"})
    ])

app.layout = html.Div(style={"fontFamily":"ui-sans-serif","padding":"16px"}, children=[
    html.H1("Proyecto Final — Dashboard interactivo", style={"marginBottom":"0"}),
    html.P("Incertidumbre (bootstrap), contraste de hipotesis (Welch) y modelos (lineal/logistico) con controles y narrativa."),

    html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4, 1fr)","gap":"12px"}, children=[
        kpi_card("Observaciones", len(df), "{:d}"),
        kpi_card("Aprobados (>=60)", int(df['aprueba'].sum()), "{:d}"),
        kpi_card("Media puntuacion", df['puntuacion'].mean()),
        kpi_card("Desv. estandar", df['puntuacion'].std())
    ]),

    html.H2("1) Exploracion y filtros"),
    html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4, 1fr)","gap":"12px","alignItems":"end"}, children=[
        html.Div([html.Label("Metodo"), dcc.Dropdown(options=[{"label":"Todos","value":"ALL"},{"label":"A","value":"A"},{"label":"B","value":"B"}], value="ALL", id="f-metodo")]),
        html.Div([html.Label("Train size"), dcc.Slider(0.5, 0.9, 0.05, value=0.7, marks=None, tooltip={"placement":"bottom"}, id="train-size")]),
        html.Div([html.Label("Ridge (alpha)"), dcc.Slider(0.0, 10.0, 0.5, value=0.0, marks=None, tooltip={"placement":"bottom"}, id="alpha")]),
        html.Div([html.Label("Logistic C (1/lambda)"), dcc.Slider(0.1, 10.0, 0.1, value=1.0, marks=None, tooltip={"placement":"bottom"}, id="C")])
    ]),

    dcc.Store(id="store-filtered"),

    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px","marginTop":"12px"}, children=[
        dcc.Graph(id="scatter-horas-score"),
        dcc.Graph(id="box-score-metodo"),
    ]),

    html.H2("2) Contraste de hipotesis y bootstrap"),
    html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4, 1fr)","gap":"12px"}, children=[
        html.Div(id="kpi-meanA"),
        html.Div(id="kpi-meanB"),
        html.Div(id="kpi-tstat"),
        html.Div(id="kpi-pvalue")
    ]),
    dcc.Graph(id="hist-bootstrap"),

    html.H2("3) Modelo lineal (prediccion de puntuacion)"),
    html.Div(style={"display":"grid","gridTemplateColumns":"repeat(2, 1fr)","gap":"12px"}, children=[
        html.Div(id="kpi-mse"),
        html.Div(id="kpi-r2")
    ]),
    dcc.Graph(id="scatter-yhat-vs-y"),

    html.H2("4) Modelo logistico (probabilidad de aprobar)"),
    html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4, 1fr)","gap":"12px"}, children=[
        html.Div(id="kpi-acc"),
        html.Div(id="kpi-sens"),
        html.Div(id="kpi-espec"),
        html.Div([html.Label("Umbral clasificacion"), dcc.Slider(0.1, 0.9, 0.05, value=0.5, id="threshold", tooltip={"placement":"bottom"})])
    ]),
    dcc.Checklist(id="feat-logistic", options=[{"label":"Incluir metodo (B dummy)","value":"metodo"}], value=[], inline=True),
    dcc.Graph(id="roc-curve"),
    dcc.Graph(id="conf-matrix"),

    html.H2("5) Narrativa y documentacion breve"),
    dcc.Markdown("""Resumen de decisiones y mejoras
- Welch t-test para comparar medias sin asumir varianzas iguales.
- Bootstrap (2000 replicas) para un IC95% de la diferencia de medias A-B.
- Lineal: OLS y opcion de Ridge (alpha) para regularizacion.
- Logistico: control de C (inverso de la regularizacion) y umbral para ajustar sensibilidad/especificidad.
- Controles de train size y seleccion de features permiten experimentar y evaluar impacto en las metricas.
""")
])

@callback(Output("store-filtered", "data"), Input("f-metodo","value"))
def filter_data(metodo):
    if metodo=="ALL":
        dff = df.copy()
    else:
        dff = df[df['metodo']==metodo].copy()
    return dff.to_json(orient="split")

@callback(Output("scatter-horas-score","figure"),
          Output("box-score-metodo","figure"),
          Input("store-filtered","data"))
def update_exploratory(data_json):
    import pandas as pd
    dff = pd.read_json(data_json, orient="split")
    fig1 = px.scatter(dff, x="horas_estudio", y="puntuacion", color="metodo", trendline="ols")
    fig1.update_layout(height=400)
    fig2 = px.box(dff, x="metodo", y="puntuacion", points="all")
    fig2.update_layout(height=400)
    return fig1, fig2

@callback(Output("kpi-meanA","children"),
          Output("kpi-meanB","children"),
          Output("kpi-tstat","children"),
          Output("kpi-pvalue","children"),
          Output("hist-bootstrap","figure"),
          Input("store-filtered","data"))
def update_inference(data_json):
    import pandas as pd
    dff = pd.read_json(data_json, orient="split")
    res_t = welch_t_test(dff)
    boot = bootstrap_mean_diff(dff)
    def card(title, v): return kpi_card(title, v)
    fig = go.Figure()
    fig.add_histogram(x=boot['diffs'], nbinsx=50)
    fig.add_vline(x=boot['ci_low'], line_dash="dash")
    fig.add_vline(x=boot['ci_high'], line_dash="dash")
    fig.update_layout(title=f"Bootstrap Delta Media (A-B) IC95% [{boot['ci_low']:.2f}, {boot['ci_high']:.2f}]")
    return (
        card("Media A", res_t['mean_A']),
        card("Media B", res_t['mean_B']),
        card("t statistic", res_t['t_stat']),
        card("p-valor", res_t['p_value']),
        fig
    )

@callback(Output("kpi-mse","children"),
          Output("kpi-r2","children"),
          Output("scatter-yhat-vs-y","figure"),
          Input("store-filtered","data"),
          Input("alpha","value"),
          Input("train-size","value"))
def update_linear(data_json, alpha, train_value):
    import pandas as pd
    dff = pd.read_json(data_json, orient="split")
    features = ["horas_estudio","horas_suenio","gpa_previo"]
    res = fit_linear(dff, features=features, alpha=alpha, test_size=1-train_value)
    fig = px.scatter(x=res['yte'], y=res['yhat'], labels={'x':'y verdadero', 'y':'y predicho'})
    fig.add_trace(go.Scatter(x=[float(np.min(res['yte'])), float(np.max(res['yte']))],
                             y=[float(np.min(res['yte'])), float(np.max(res['yte']))],
                             mode='lines', name='y=x'))
    return kpi_card("MSE", res['MSE']), kpi_card("R2", res['R2']), fig

@callback(Output("kpi-acc","children"),
          Output("kpi-sens","children"),
          Output("kpi-espec","children"),
          Output("roc-curve","figure"),
          Output("conf-matrix","figure"),
          Input("store-filtered","data"),
          Input("C","value"),
          Input("threshold","value"),
          Input("feat-logistic","value"),
          Input("train-size","value"))
def update_logistic(data_json, C, threshold, features, train_value):
    import pandas as pd
    dff = pd.read_json(data_json, orient="split")
    feats = ('horas_estudio',)
    if 'metodo' in features:
        feats = ('metodo','horas_estudio')
    res = fit_logistic(dff, features=feats, C=C, threshold=threshold, test_size=1-train_value)
    roc = go.Figure()
    roc.add_trace(go.Scatter(x=res['fpr'], y=res['tpr'], mode='lines', name=f"ROC AUC={res['roc_auc']:.3f}"))
    roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Azar', line=dict(dash='dash')))
    roc.update_layout(xaxis_title="FPR", yaxis_title="TPR")
    cm = confusion_matrix(res['yte'], res['yhat'])
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=['Pred 0','Pred 1'], y=['Real 0','Real 1'], text=cm, texttemplate="%{text}", showscale=False))
    cm_fig.update_layout(title="Matriz de confusion")
    return (kpi_card("Accuracy", res['acc']),
            kpi_card("Sensibilidad", res['sens']),
            kpi_card("Especificidad", res['espec']),
            roc, cm_fig)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
