# Nasdaq - Microsoft Stock

Pasta da base MSFT do trabalho de Séries Temporais.

## Estrutura

- `main.ipynb`: análise completa da base.
- `dataset`: MSFT, NASDAQ Composite e VIX.
- `resultados`: métricas, previsões, resíduos e importância das features.
- `analise.html`: relatório paginado e autocontido para apresentação.
- `analise.pdf`: versão final em PDF com o mesmo conteúdo do HTML.
- `registro_demandas.csv`: atividades realizadas por Gabriel Bento.

## Execução

```powershell
python -m pip install -r requirements.txt
jupyter notebook main.ipynb
```

O target é `Open`. Os modelos avaliados são SARIMAX, Holt-Winters, Random Forest e XGBoost.

Esta pasta contém a entrega completa da base MSFT. A contagem de vitórias e a posição média entre bases devem ser calculadas no relatório consolidado quando os resultados das outras quatro bases estiverem disponíveis.
