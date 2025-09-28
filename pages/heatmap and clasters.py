import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Завантаження даних
df = pd.read_csv("./WHR_all2.csv")

st.title("Heatmap та кластеризація індексів")

# --- Вибір індекса для Heatmap ---
st.subheader("Карта країн за індексом")

heatmap_index = st.selectbox(
    "Оберіть індекс для heatmap:",
    [
        "Ladder score",
        "AI_index_base",
        "AI_index_y_region",
        "AI_index_new_features",
        "AI_index_new_features_y_reg",
        "Autoencoder_Index_base",
        "Autoencoder_Index_new_features",
        "VAE_base",
        "VAE_new_features"
    ]
)

# Вибір року
year = st.slider("Оберіть рік:", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].min()))

df_year = df[df["Year"] == year]

# Мінімум і максимум для кольорової шкали
vmin, vmax = df[heatmap_index].min(), df[heatmap_index].max()

fig = px.choropleth(
    df_year,
    locations="Country",
    color=heatmap_index,
    locationmode="country names",
    template="plotly_dark",
    color_continuous_scale="YlOrBr",
    range_color=[0, 10],
    title=f"{heatmap_index} у {year} році"
)
st.plotly_chart(fig, use_container_width=True)


# --- Кластеризація ---
st.subheader("Кластеризація індексів")

cluster_index = st.selectbox(
    "Оберіть індекс для кластеризації:",
    [
        "AI_index_base",
        "AI_index_y_region",
        "AI_index_new_features",
        "AI_index_new_features_y_reg",
        "Autoencoder_Index_base",
        "Autoencoder_Index_new_features",
        "VAE_base",
        "VAE_new_features"
    ]
)

clusters_n = st.slider("Кількість кластерів:", 2, 9, 3)
cluster_year = st.slider("Оберіть рік для кластеризації:", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].min()))

df_cluster = df[df["Year"] == cluster_year][["Country", cluster_index, "Ladder score"]].dropna()

if len(df_cluster) >= clusters_n:
    kmeans = KMeans(n_clusters=clusters_n, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(df_cluster[[cluster_index]])

    # Коефіцієнт силуету
    sil_score = silhouette_score(df_cluster[[cluster_index]], df_cluster["Cluster"])
    st.write(f"**Коефіцієнт силуетів:** {sil_score:.3f}")

    # Візуалізація у 2D
    if "Ladder score" in df_cluster.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            df_cluster[cluster_index], 
            df_cluster["Ladder score"],
            c=df_cluster["Cluster"], cmap="spring", s=80
        )

        ax.set_title(f"Кластеризація за {cluster_index} ({cluster_year})")
        ax.set_xlabel(cluster_index)
        ax.set_ylabel("Ladder score")

        legend1 = ax.legend(*scatter.legend_elements(), title="Кластер")
        ax.add_artist(legend1)

        st.pyplot(fig)
    else:
        st.warning("Немає колонки 'Ladder score' для побудови 2D-графіка.")

else:
    st.warning("Недостатньо даних для кластеризації у цьому році.")

