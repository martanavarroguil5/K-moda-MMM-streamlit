import matplotlib.pyplot as plt
import pandas as pd

# Datos de métricas de modelos
model_data = {
    "Modelo": [
        "NaiveRolling4",
        "ElasticNetControls",
        "ElasticNetRawMedia",
        "ElasticNetTransformedMedia"
    ],
    "MAPE (test)": [20.34, 19.19, 17.62, 17.42],
    "MAE (test)": [456.40, 423.07, 397.04, 392.30],
    "RMSE (test)": [685.65, 579.99, 551.29, 549.29],
    "Bias (test)": [54.18, 129.24, 71.99, 54.71]
}
df = pd.DataFrame(model_data)

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#cccccc']*len(df.columns)
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.5)
plt.tight_layout()
plt.savefig('reports/figures/5_modelado/model_comparison_table.png', dpi=200, bbox_inches='tight')
plt.close()
