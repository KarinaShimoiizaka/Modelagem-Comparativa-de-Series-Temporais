# Microsoft Stock — Séries Temporais

Pasta da base MSFT do trabalho de Séries Temporais.

## Estrutura

- `main.ipynb`: análise completa e executada da base.
- `dataset`: série histórica da Microsoft.
- `resultados`: métricas, previsões, resíduos, coeficientes e importâncias.
- `analise.html`: relatório paginado para apresentação.
- `analise.pdf`: versão final em PDF com o mesmo conteúdo do HTML.
- `registro_demandas.csv`: atividades realizadas por Gabriel Bento.

## Execução

```powershell
python -m pip install -r requirements.txt
jupyter notebook main.ipynb
```

O target é `Open`. As variáveis exógenas são `Close` e `Volume`, usadas com defasagem de um pregão para evitar vazamento. Os modelos avaliados são SARIMAX, Holt-Winters, Random Forest e XGBoost.

Somente `dataset/Microsoft_Stock.csv` participa da análise. Os arquivos `NASDAQ.csv` e `VIX.csv` foram preservados na pasta por histórico do projeto, mas não entram em nenhuma feature ou modelo.

Na execução final, o SARIMAX obteve o menor MAE (1,643 USD) e o XGBoost ficou em segundo lugar (2,583 USD). A contagem de vitórias e a posição média entre bases devem ser calculadas no relatório consolidado quando os resultados das outras quatro bases estiverem disponíveis.
