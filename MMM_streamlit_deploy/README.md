# Deploy de Streamlit

Esta carpeta contiene lo necesario para desplegar la app de Streamlit sin incluir el proyecto completo de modelado.

## Contenido

- `streamlit_app.py`: app principal.
- `.streamlit/config.toml`: tema visual de Streamlit.
- `src/`: modulo Python usado por la app.
- `reports/figures`: imagenes y artes que muestra la app.
- `reports/tables`: tablas CSV y JSON consumidas por la app.
- `data/processed`: JSON minimos que usa la app para resumenes y metricas.
- `requirements.txt`: dependencias para ejecutar la app.

## Arranque local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notas

- Esta carpeta esta pensada para ejecutar la app, no para recalcular modelos ni regenerar reportes.
- Si quieres desplegarla en Streamlit Community Cloud, sube esta carpeta tal cual y usa `streamlit_app.py` como entrypoint.
- Si la vas a desplegar en otro entorno, asegúrate de que el directorio de trabajo sea la raiz de esta carpeta.
