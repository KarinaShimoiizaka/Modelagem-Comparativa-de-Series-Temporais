"""Gera as figuras (img/) e os números (resultados_relatorio.json) usados em analise.html.

Reproduz o pipeline dos notebooks 1-5 sobre data/Pilgrim_s_Pride_Corporation_2.csv,
com os hiperparâmetros já escolhidos e registrados em all_modelos.json (não refaz
auto_arima nem Optuna). Também calcula diagnósticos que os notebooks não salvam:
qualidade dos dados, STL, baselines no mesmo protocolo walk-forward e resíduos.

Uso: python gerar_relatorio.py   (rodar de dentro da pasta pilgrims/)
"""
import json
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from xgboost import XGBRegressor

from model_utils import calculate_metrics, evaluate_baselines, walk_forward_predict

warnings.filterwarnings("ignore")
os.makedirs("img", exist_ok=True)

# Estilo dos gráficos exportados para o relatório (mesmo da base de temperatura)
COR = "#2f5d50"
COR_EXT = "#9a4b3a"
CINZA = "#9a958a"
CORES_MODELOS = {
    "SARIMAX": "#2f5d50",
    "Holt-Winters aditivo": "#c08a3e",
    "Holt-Winters multiplicativo": "#6b6558",
    "Random Forest": "#3e6f9a",
    "XGBoost": "#9a4b3a",
}
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e3da",
    "axes.titlelocation": "left",
})

TARGET = "Close"
DATE_COLUMN = "Date"
EXOGS = ["TSN_Close", "XLP_Close", "CALM_Close"]
M = 76
STEP = 50
R = {}  # todos os números do relatório


def salvar(fig, nome):
    fig.savefig(f"img/{nome}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def residuo_info(real, previsto):
    res = pd.Series(np.asarray(real) - np.asarray(previsto), index=real.index).dropna()
    lb = acorr_ljungbox(res, lags=[10, 20], return_df=True)
    return res, {
        "vies": float(res.mean()),
        "dp": float(res.std()),
        "acf1": float(res.autocorr(1)),
        "lb10": float(lb.loc[10, "lb_stat"]), "lb10_p": float(lb.loc[10, "lb_pvalue"]),
        "lb20": float(lb.loc[20, "lb_stat"]), "lb20_p": float(lb.loc[20, "lb_pvalue"]),
        "pct_subestimado": float((res > 0).mean() * 100),
    }


# ======================================================================
# 1. Dados — mesma preparação dos notebooks
# ======================================================================
bruto = pd.read_csv("data/Pilgrim_s_Pride_Corporation_2.csv")
bruto[DATE_COLUMN] = pd.to_datetime(bruto[DATE_COLUMN], errors="coerce", utc=True).dt.tz_localize(None)
bruto = bruto.set_index(DATE_COLUMN).sort_index()
df = bruto.drop(columns=["Open", "High", "Low", "Volume", "USD_MXN"])

original = pd.read_csv("data/Pilgrim_s_Pride_Corporation.csv")
original[DATE_COLUMN] = pd.to_datetime(original[DATE_COLUMN], utc=True).dt.tz_localize(None).dt.normalize()
original = original.set_index(DATE_COLUMN)

feriados = bruto.index.difference(original.index)
dias_uteis = pd.bdate_range(bruto.index[0], bruto.index[-1])
R["dados"] = {
    "n": len(df), "inicio": str(df.index[0].date()), "fim": str(df.index[-1].date()),
    "ausentes": int(bruto.isna().sum().sum()),
    "datas_duplicadas": int(bruto.index.duplicated().sum()),
    "dias_uteis_faltando": len(dias_uteis.difference(bruto.index)),
    "linhas_feriado_preenchidas": len(feriados),
    "feriado_volume_zero": int((bruto.loc[feriados, "Volume"] == 0).sum()),
    "feriado_close_repetido": int((bruto.loc[feriados, "Close"] == bruto[TARGET].shift(1).loc[feriados]).sum()),
    "dividendos_periodo": original.loc[original.index >= bruto.index[0]].query("Dividends > 0")["Dividends"].to_dict(),
    "close_desc": df[TARGET].describe().to_dict(),
    "close_anual": df[TARGET].resample("YE").agg(["mean", "min", "max"]).round(2).to_dict("index"),
}
R["dados"]["dividendos_periodo"] = {str(k.date()): v for k, v in R["dados"]["dividendos_periodo"].items()}
R["dados"]["close_anual"] = {str(k.year): v for k, v in R["dados"]["close_anual"].items()}

retorno = df[TARGET].diff().dropna()
q1, q3 = retorno.quantile([0.25, 0.75])
iqr = q3 - q1
R["dados"]["outliers_retorno_iqr"] = int(((retorno < q1 - 1.5 * iqr) | (retorno > q3 + 1.5 * iqr)).sum())
R["dados"]["maiores_quedas"] = {str(k.date()): float(v) for k, v in retorno.nsmallest(5).items()}
R["dados"]["maiores_altas"] = {str(k.date()): float(v) for k, v in retorno.nlargest(5).items()}
for col in [TARGET] + EXOGS:
    serie = df[col]
    a, b = serie.quantile([0.25, 0.75])
    R["dados"][f"outliers_nivel_{col}"] = int(((serie < a - 1.5 * (b - a)) | (serie > b + 1.5 * (b - a))).sum())

# Split igual aos notebooks 1 e 2: últimos 25% para teste
h = int(len(df) * 0.25)
train_original, test_original = df.iloc[:-h], df.iloc[-h:]
R["split"] = {
    "treino": len(train_original), "teste": len(test_original),
    "treino_inicio": str(train_original.index[0].date()), "treino_fim": str(train_original.index[-1].date()),
    "teste_inicio": str(test_original.index[0].date()), "teste_fim": str(test_original.index[-1].date()),
}

# Correlações no treino: nível e 1ª diferença
R["correlacao"] = {
    "nivel": train_original.corr()[TARGET].drop(TARGET).to_dict(),
    "diferenca": train_original.diff().corr()[TARGET].drop(TARGET).to_dict(),
    "volume_mxn_diferenca": bruto.iloc[:-h][[TARGET, "Volume", "USD_MXN"]].diff().corr()[TARGET].drop(TARGET).to_dict(),
}

# ======================================================================
# 2. Gráficos da série e das externas
# ======================================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df[TARGET], color=COR, lw=0.9, label="Close")
ax.plot(df.index, df[TARGET].rolling(M).mean(), color=COR_EXT, lw=1.2, label=f"média móvel {M} pregões")
ax.axvline(test_original.index[0], color=CINZA, ls="--", lw=1)
ax.text(test_original.index[0], ax.get_ylim()[1] * 0.97, "  início do teste", color=CINZA, va="top", fontsize=9)
ax.set_title("Pilgrim's Pride (PPC) — preço de fechamento ajustado, US$")
ax.set_ylabel("US$")
ax.legend(frameon=False, loc="upper left")
salvar(fig, "serie_alvo.png")

fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
nomes = {"TSN_Close": "Tyson Foods (TSN)", "XLP_Close": "Consumer Staples Select Sector SPDR (XLP)",
         "CALM_Close": "Cal-Maine Foods (CALM)"}
for ax, col in zip(axes, EXOGS):
    ax.plot(df.index, df[col], color=COR, lw=0.9)
    ax.set_title(f"{nomes[col]} — fechamento, US$", fontsize=10)
    ax.axvline(test_original.index[0], color=CINZA, ls="--", lw=1)
salvar(fig, "externas.png")

# ======================================================================
# 3. Estacionariedade, ACF/PACF e STL
# ======================================================================
def testes(serie):
    adf = adfuller(serie.dropna())
    kp = kpss(serie.dropna(), regression="c", nlags="auto")
    return {"adf_stat": float(adf[0]), "adf_p": float(adf[1]), "kpss_stat": float(kp[0]), "kpss_p": float(kp[1])}


R["estacionariedade"] = {"nivel": testes(df[TARGET]), "diferenca": testes(df[TARGET].diff())}

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
plot_acf(df[TARGET].diff().dropna(), lags=40, ax=axes[0], title="ACF — 1ª diferença do Close", color=COR, vlines_kwargs={"colors": COR})
plot_pacf(df[TARGET].diff().dropna(), lags=40, ax=axes[1], title="PACF — 1ª diferença do Close", color=COR, vlines_kwargs={"colors": COR}, method="ywm")
for ax in axes:
    ax.set_ylim(-0.15, 0.15)
salvar(fig, "acf_pacf.png")
acf_dif = [float(df[TARGET].diff().autocorr(k)) for k in range(1, 11)]
R["acf_diferenca_1a10"] = acf_dif

# Varredura de m na 1ª diferença padronizada (célula comentada do notebook 1)
dif_pad = (df[TARGET].diff() - df[TARGET].diff().mean()) / df[TARGET].diff().std()
dif_pad = dif_pad.dropna()
varredura = []
for m in range(2, 101):
    s = STL(dif_pad, period=m, robust=True).fit()
    varredura.append((m, float(max(0, 1 - np.var(s.resid) / np.var(s.seasonal + s.resid)))))
varredura = pd.Series(dict(varredura))
R["stl_varredura"] = {
    "m_max": int(varredura.idxmax()), "forca_max": float(varredura.max()),
    "forca_m76": float(varredura.loc[M]), "forca_m5": float(varredura.loc[5]),
    "forca_m21": float(varredura.loc[21]), "forca_m63": float(varredura.loc[63]),
    "forca_min": float(varredura.min()), "forca_mediana": float(varredura.median()),
    "top5": {int(k): float(v) for k, v in varredura.nlargest(5).items()},
}
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.plot(varredura.index, varredura.values, color=COR, lw=1.2, marker="o", ms=2.5)
ax.axvline(M, color=COR_EXT, ls="--", lw=1)
ax.text(M + 1, varredura.max(), f"m = {M}", color=COR_EXT, va="top", fontsize=9)
ax.set_title("Força da sazonalidade do STL na 1ª diferença padronizada, por período m")
ax.set_xlabel("período m (pregões)")
ax.set_ylabel("F_S")
salvar(fig, "stl_varredura_m.png")

# STL no nível (preço), período 76
stl = STL(df[TARGET], period=M, robust=True).fit()
forca_t = max(0, 1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))
forca_s = max(0, 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid))
tend = stl.trend
R["stl_nivel"] = {
    "forca_tendencia": float(forca_t), "forca_sazonalidade": float(forca_s),
    "sazonal_amp": float(stl.seasonal.max() - stl.seasonal.min()),
    "sazonal_dp": float(stl.seasonal.std()), "resid_dp": float(stl.resid.std()),
    "resid_dp_ano": stl.resid.groupby(stl.resid.index.year).std().round(3).to_dict(),
    "sazonal_dp_ano": stl.seasonal.groupby(stl.seasonal.index.year).std().round(3).to_dict(),
    "tendencia_anual": tend.resample("YE").last().round(2).rename(lambda x: x.year).to_dict(),
    "tendencia_min": [str(tend.idxmin().date()), float(tend.min())],
    "tendencia_max": [str(tend.idxmax().date()), float(tend.max())],
    "resid_adf_p": float(adfuller(stl.resid)[1]),
    "resid_out_3dp": int((stl.resid.abs() > 3 * stl.resid.std()).sum()),
}
fig, axes = plt.subplots(4, 1, figsize=(12, 8.5), sharex=True)
for ax, (nome, serie) in zip(axes[:3], [("Observado", stl.observed), ("Tendência", stl.trend), ("Sazonal (m = 76)", stl.seasonal)]):
    ax.plot(serie.index, serie, color=COR, lw=0.9)
    ax.set_title(nome, fontsize=10)
axes[3].scatter(stl.resid.index, stl.resid, s=3, color=COR)
axes[3].set_title("Resíduo", fontsize=10)
salvar(fig, "stl.png")

# ======================================================================
# 4. Protocolo walk-forward (figura)
# ======================================================================
fig, ax = plt.subplots(figsize=(12, 3.4))
ax.plot(train_original.index, train_original[TARGET], color=CINZA, lw=0.9, label=f"treino inicial ({len(train_original)} pregões)")
ax.plot(test_original.index, test_original[TARGET], color=COR, lw=0.9, label=f"teste ({len(test_original)} pregões)")
origens = test_original.index[::STEP]
for i, o in enumerate(origens):
    ax.axvline(o, color=COR_EXT, lw=0.6, alpha=0.6, label="origens de reajuste (a cada 50)" if i == 0 else None)
ax.set_title("Janela expansiva: reajuste a cada 50 pregões no bloco de teste")
ax.set_ylabel("US$")
ax.legend(frameon=False, loc="upper left", fontsize=9)
salvar(fig, "walk_forward.png")

# ======================================================================
# 5. Modelos — mesmos hiperparâmetros registrados em all_modelos.json
# ======================================================================
with open("all_modelos.json", encoding="utf-8") as f:
    registrados = {r["modelo"]: r for r in json.load(f)}

y_test = test_original[TARGET]
previsoes = {}
tempos = {}

# --- SARIMAX (notebook 1): escalonamento no treino, exógenas do próprio dia
scaler = StandardScaler().fit(train_original)
train = pd.DataFrame(scaler.transform(train_original), index=train_original.index, columns=train_original.columns)
test = pd.DataFrame(scaler.transform(test_original), index=test_original.index, columns=test_original.columns)
pos = list(train.columns).index(TARGET)
ordem = tuple(registrados["Sarimax"]["params"]["order"])
sazonal = tuple(registrados["Sarimax"]["params"]["seasonal_order"])


def sarimax_wf(exogs):
    historico = train.copy()
    saida = []
    for inicio in range(0, len(test), STEP):
        fim = min(inicio + STEP, len(test))
        mod = SARIMAX(historico[TARGET], exog=historico[exogs] if exogs else None, order=ordem,
                      seasonal_order=sazonal, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
        saida.extend(mod.forecast(steps=fim - inicio, exog=test[exogs].iloc[inicio:fim] if exogs else None).to_numpy())
        historico = pd.concat([historico, test.iloc[inicio:fim]])
    return pd.Series(np.asarray(saida) * scaler.scale_[pos] + scaler.mean_[pos], index=test.index)


t0 = time.time()
previsoes["SARIMAX"] = sarimax_wf(EXOGS)
tempos["SARIMAX"] = time.time() - t0
t0 = time.time()
previsao_arima = sarimax_wf([])
tempos["ARIMA sem exógenas"] = time.time() - t0

ajuste = SARIMAX(train[TARGET], exog=train[EXOGS], order=ordem, seasonal_order=sazonal,
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=100)
R["sarimax_coef"] = {
    nome: {"coef": float(ajuste.params[nome]), "ep": float(ajuste.bse[nome]), "p": float(ajuste.pvalues[nome])}
    for nome in ajuste.params.index
}
R["sarimax_aic"] = float(ajuste.aic)
lb_treino = acorr_ljungbox(ajuste.resid.iloc[1:], lags=[10, 20], return_df=True)
R["sarimax_lb_treino"] = {"lb10_p": float(lb_treino.loc[10, "lb_pvalue"]), "lb20_p": float(lb_treino.loc[20, "lb_pvalue"])}

# --- Holt-Winters (notebook 2)
def hw_wf(sazonalidade):
    historico = train_original[TARGET].copy()
    saida = []
    for inicio in range(0, len(y_test), STEP):
        fim = min(inicio + STEP, len(y_test))
        desloc = max(0.0, -float(historico.min()) + 1e-6) if sazonalidade == "mul" else 0.0
        mod = ExponentialSmoothing(historico + desloc, trend="add", seasonal=sazonalidade, seasonal_periods=M,
                                   initialization_method="heuristic").fit()
        saida.extend((mod.forecast(fim - inicio) - desloc).to_numpy())
        historico = pd.concat([historico, y_test.iloc[inicio:fim]])
    return pd.Series(saida, index=y_test.index)


t0 = time.time()
previsoes["Holt-Winters aditivo"] = hw_wf("add")
tempos["Holt-Winters aditivo"] = time.time() - t0
t0 = time.time()
previsoes["Holt-Winters multiplicativo"] = hw_wf("mul")
tempos["Holt-Winters multiplicativo"] = time.time() - t0

hw_final = ExponentialSmoothing(train_original[TARGET], trend="add", seasonal="add", seasonal_periods=M,
                                initialization_method="heuristic").fit()
R["hw_aditivo_treino"] = {k: float(hw_final.params[k]) for k in ["smoothing_level", "smoothing_trend", "smoothing_seasonal"]}
R["hw_aditivo_treino"]["aic"] = float(hw_final.aic)
fig, axes = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
for ax, (nome, serie) in zip(axes, [("Nível (l_t)", hw_final.level), ("Tendência (b_t)", hw_final.trend), ("Sazonal (s_t), m = 76", hw_final.season)]):
    ax.plot(train_original.index, serie, color=COR, lw=0.9)
    ax.set_title(nome, fontsize=10)
salvar(fig, "componentes_hw.png")

# --- Random Forest e XGBoost (notebooks 3 e 4): mesmas 85 features
historico = df[EXOGS + [TARGET]].shift(1)
X = pd.DataFrame(index=df.index)
for coluna in historico.columns:
    for periodo in [1, 2, 3, 7, 14, 30, M, 2 * M]:
        X[f"{coluna}_lag_{periodo}"] = historico[coluna].shift(periodo - 1)
for coluna in historico.columns:
    X[f"{coluna}_delta_1"] = historico[coluna] - historico[coluna].shift(1)
    X[f"{coluna}_delta_sazonal"] = historico[coluna] - historico[coluna].shift(M - 1)
for janela in [3, 7, 14, 30, M]:
    for coluna in historico.columns:
        X[f"{coluna}_media_{janela}"] = historico[coluna].rolling(janela).mean()
        X[f"{coluna}_desvio_{janela}"] = historico[coluna].rolling(janela).std()
alvo_hist = historico[TARGET]
X[f"{TARGET}_delta_3"] = alvo_hist - alvo_hist.shift(3)
X[f"{TARGET}_delta_7"] = alvo_hist - alvo_hist.shift(7)
X[f"{TARGET}_delta_30"] = alvo_hist - alvo_hist.shift(30)
X[f"{TARGET}_media_sazonal"] = alvo_hist.rolling(M).mean()
X[f"{TARGET}_desvio_sazonal"] = alvo_hist.rolling(M).std()
base = pd.concat([X, df[TARGET]], axis=1).dropna()
X, y = base.drop(columns=[TARGET]), base[TARGET]
h_mod = int(len(X) * 0.25)
X_train, X_test, y_train, y_test_arv = X.iloc[:-h_mod], X.iloc[-h_mod:], y.iloc[:-h_mod], y.iloc[-h_mod:]
R["arvores"] = {
    "n_features": X.shape[1], "linhas": len(X), "warmup_descartado": len(df) - len(X),
    "primeira_linha": str(X.index[0].date()), "treino": len(X_train), "teste": len(X_test),
    "teste_inicio": str(X_test.index[0].date()),
    "treino_fim": str(X_train.index[-1].date()),
}

# Checagem de vazamento: alterar o dia t não muda nenhuma feature da linha t
dia = df.index[2000]
df_alt = df.copy()
df_alt.loc[dia] = df_alt.loc[dia] * 10
hist_alt = df_alt[EXOGS + [TARGET]].shift(1)
R["arvores"]["checagem_vazamento_lag1"] = bool(hist_alt.loc[dia].equals(historico.loc[dia]))

rf_params = registrados["RandomForest"]["params"]
xgb_params = registrados["XGBoost"]["params"]
t0 = time.time()
previsoes["Random Forest"] = pd.Series(walk_forward_predict(
    lambda: RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1), X_train, y_train, X_test, y_test_arv, step=STEP),
    index=y_test_arv.index)
tempos["Random Forest"] = time.time() - t0
t0 = time.time()
previsoes["XGBoost"] = pd.Series(walk_forward_predict(
    lambda: XGBRegressor(**xgb_params, objective="reg:squarederror", eval_metric="mae", random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test_arv, step=STEP), index=y_test_arv.index)
tempos["XGBoost"] = time.time() - t0

# ======================================================================
# 6. Métricas, baselines e resíduos
# ======================================================================
naive_bloco = pd.Series(np.nan, index=y_test.index)
valores = pd.concat([train_original[TARGET], y_test])
for inicio in range(0, len(y_test), STEP):
    fim = min(inicio + STEP, len(y_test))
    naive_bloco.iloc[inicio:fim] = valores.iloc[len(train_original) + inicio - 1]
persistencia = df[TARGET].shift(1).loc[y_test.index]

comum = y_test_arv.index  # janela em que todos os modelos têm previsão
R["metricas"] = {}
R["residuos"] = {}
residuos = {}
for nome, prev in previsoes.items():
    real = y_test.loc[prev.index]
    R["metricas"][nome] = calculate_metrics(real, prev)
    R["metricas"][nome]["mae_janela_comum"] = float(np.mean(np.abs(y_test.loc[comum] - prev.loc[comum])))
    R["metricas"][nome]["n"] = len(prev)
    R["metricas"][nome]["tempo_s"] = tempos[nome]
    residuos[nome], R["residuos"][nome] = residuo_info(real, prev)
R["metricas"]["ARIMA(2,1,2) sem exógenas"] = calculate_metrics(y_test, previsao_arima)
R["metricas"]["ARIMA(2,1,2) sem exógenas"]["mae_janela_comum"] = float(np.mean(np.abs(y_test.loc[comum] - previsao_arima.loc[comum])))
_, R["residuos"]["ARIMA(2,1,2) sem exógenas"] = residuo_info(y_test, previsao_arima)

R["baselines"] = {
    "notebook": evaluate_baselines(train_original[TARGET], y_test, M),
    "naive_por_bloco": calculate_metrics(y_test, naive_bloco),
    "naive_por_bloco_janela_comum": float(np.mean(np.abs(y_test.loc[comum] - naive_bloco.loc[comum]))),
    "persistencia_1_passo": calculate_metrics(y_test, persistencia),
    "persistencia_1_passo_janela_comum": calculate_metrics(y_test.loc[comum], persistencia.loc[comum]),
}
_, R["residuos"]["Persistência (1 passo)"] = residuo_info(y_test.loc[comum], persistencia.loc[comum])
_, R["residuos"]["Naive por bloco"] = residuo_info(y_test, naive_bloco)

# Erro por passo dentro do bloco (1 a 50 pregões após a origem)
passo = pd.Series(np.arange(len(y_test)) % STEP + 1, index=y_test.index)
faixas = {"1-5": (1, 5), "6-20": (6, 20), "21-35": (21, 35), "36-50": (36, 50)}
R["mae_por_passo"] = {}
for nome, prev in [("SARIMAX", previsoes["SARIMAX"]), ("ARIMA sem exógenas", previsao_arima),
                   ("Holt-Winters aditivo", previsoes["Holt-Winters aditivo"]), ("Naive por bloco", naive_bloco)]:
    erro = (y_test - prev).abs()
    R["mae_por_passo"][nome] = {k: float(erro[(passo >= a) & (passo <= b)].mean()) for k, (a, b) in faixas.items()}

# MAE por ano no teste
R["mae_por_ano"] = {}
for nome, prev in list(previsoes.items()) + [("Persistência (1 passo)", persistencia.loc[comum])]:
    erro = (y_test.loc[prev.index] - prev).abs()
    R["mae_por_ano"][nome] = erro.groupby(erro.index.year).mean().round(3).rename(str).to_dict()

R["teste_desc"] = {
    "min": [str(y_test.idxmin().date()), float(y_test.min())], "max": [str(y_test.idxmax().date()), float(y_test.max())],
    "max_treino": float(train_original[TARGET].max()), "max_treino_arvores": float(y_train.max()),
    "dias_acima_max_treino_arvores": int((y_test_arv > y_train.max()).sum()),
    "dias_acima_max_treino": int((y_test > train_original[TARGET].max()).sum()),
}

# ======================================================================
# 7. Figuras de resultados
# ======================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, grupo, titulo in [
    (axes[0], ["SARIMAX", "Holt-Winters aditivo", "Holt-Winters multiplicativo"], "Modelos estatísticos — previsão de até 50 pregões a partir de cada origem"),
    (axes[1], ["Random Forest", "XGBoost"], "Modelos de árvore — previsão 1 pregão à frente (features até a véspera)"),
]:
    ax.plot(y_test.index, y_test, color="black", lw=1.6, label="real")
    for nome in grupo:
        ax.plot(previsoes[nome].index, previsoes[nome], color=CORES_MODELOS[nome], lw=1, ls="--",
                label=f"{nome} (MAE {R['metricas'][nome]['mae']:.3f})")
    ax.set_title(titulo, fontsize=10)
    ax.set_ylabel("US$")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
salvar(fig, "previsoes_teste.png")

fig, axes = plt.subplots(len(previsoes), 2, figsize=(12, 2.4 * len(previsoes)), gridspec_kw={"width_ratios": [3, 1]})
for i, (nome, res) in enumerate(residuos.items()):
    axes[i, 0].bar(res.index, res.values, width=1.5, color=CORES_MODELOS[nome])
    axes[i, 0].axhline(res.mean(), color="black", lw=1, ls="--")
    axes[i, 0].set_title(f"{nome} — resíduo (real − previsto), US$", fontsize=10)
    plot_acf(res, lags=20, ax=axes[i, 1], title="ACF", color=CORES_MODELOS[nome], vlines_kwargs={"colors": CORES_MODELOS[nome]})
    axes[i, 1].set_ylim(-0.3, 1.05)
fig.tight_layout()
salvar(fig, "residuos.png")

for chave, nome, arquivo in [("RandomForest", "Random Forest", "importancia_random_forest.png"), ("XGBoost", "XGBoost", "importancia_xgboost.png")]:
    shap_imp = pd.Series(registrados[chave]["shap_importance"]).head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    cores = [COR_EXT if c.startswith(("TSN", "XLP", "CALM")) else COR for c in shap_imp.index]
    ax.barh(shap_imp.index, shap_imp.values, color=cores)
    ax.set_xscale("log")
    ax.set_title(f"{nome} — média de |SHAP| (US$), top 15, escala log")
    ax.set_xlabel("média de |SHAP value| (escala log)")
    salvar(fig, arquivo)

fig, ax = plt.subplots(figsize=(10, 3.8))
ordem_mae = sorted(previsoes, key=lambda n: R["metricas"][n]["mae"])
ax.barh(ordem_mae[::-1], [R["metricas"][n]["mae"] for n in ordem_mae[::-1]], color=[CORES_MODELOS[n] for n in ordem_mae[::-1]])
ax.axvline(R["baselines"]["persistencia_1_passo"]["mae"], color="black", ls=":", lw=1.2, label="persistência 1 passo")
ax.axvline(R["baselines"]["naive_por_bloco"]["mae"], color=CINZA, ls="--", lw=1.2, label="naive por bloco de 50")
ax.set_title("MAE no teste walk-forward (US$)")
ax.legend(frameon=False, fontsize=9, loc="lower right")
salvar(fig, "mae_modelos.png")

# ======================================================================
# 8. Saídas
# ======================================================================
saida = pd.DataFrame({"real": y_test, "naive_por_bloco": naive_bloco, "persistencia": persistencia,
                      "arima_sem_exogenas": previsao_arima})
for nome, prev in previsoes.items():
    saida[nome] = prev
saida.to_csv("previsoes_teste.csv", index_label="Date")
with open("resultados_relatorio.json", "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=float)
print(json.dumps(R["metricas"], indent=1, default=float))
