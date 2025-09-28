import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# Заголовок сторінки
st.set_page_config(page_title="Happiness Data Explorer", layout="wide")
st.title("Дослідження рівня щастя")

# Завантаження даних
@st.cache_data
def load_data():
    return pd.read_csv("WHR_all2.csv")

df = load_data()

# Відображення таблиці
st.subheader("Таблиця даних")

columns_to_show = [
    "Country", "Year", "Regional indicator", "Ladder score",
    "Healthy life expectancy", "Log GDP per capita", "Social support",
    "Perceptions of corruption", "Freedom to make life choices", "Generosity",
    "Freedom_trust_index", "Generosity_normalized", "Life_freedom",
    "Life_support", "Life_freedom_support", "Delta_Ladder_score",
    "AI_index_base", "AI_index_y_region", "AI_index_new_features",
    "AI_index_new_features_y_reg", "Autoencoder_Index_base",
    "Autoencoder_Index_new_features", "VAE_base", "VAE_new_features"
]

st.dataframe(df[columns_to_show])

# Пояснення до колонок
with st.expander("Пояснення до колонок"):
    st.markdown("""
    **Country** — назва країни  
    **Year** — рік  
    **Regional indicator** — назва регіону  
    **Ladder score** — середня по країні відповідь респондентів на запитання GWP «Будь ласка, уявіть собі драбину, сходинки якої пронумеровані від 0 внизу до 10 вгорі. Верхня сходинка являє собою найкраще можливе життя для Вас, а нижня – найгірше можливе життя для Вас. На якій сходинці драбини, на вашу думку, ви знаходитесь у даний момент?»
                  
    **Healthy life expectancy** — очікувана тривалість здорового життя  
    **Log GDP per capita** — логарифм ВВП на душу населення  
    **Social support** — рівень соціальної підтримки (середня відповідь (1 – так, 0 – ні) на питання “Якби ви мали якісь проблеми, чи маєте ви друзів або родичів, що можуть вам допомогти”)
                  
    **Perceptions of corruption** — середнє значення бінарних відповідей на два запитання GWP: «Чи поширена корупція в уряді чи ні?» та «Чи поширена корупція в бізнесі чи ні?»
                  
    **Freedom to make life choices** — свобода прийняття життєвих рішень (середня відповідь (1 – так, 0 – ні) на питання “Чи задоволені ви поточним рівнем вашої свободи прийняття життєвих рішень”)
                  
    **Generosity** — залишок регресійного обчислення середнього значення відповідей GWP на запитання “Чи жертвували ви гроші на благодійність протягом останнього місяця?” за логарифмом ВВП на душу населення
                  
    **Freedom_trust_index** = Freedom to make life choices × (1 - Perceptions of corruption) - зв'язок високої свободи прийняття життєвих рішень з відсутністю корупції
                 
    **Generosity_normalized** = Generosity / mean(Generosity in region) - деякі країни виглядають "щедрими", але якщо порівняти з регіональним рівнем, це може показати реальні відмінності
                  
    **Life_freedom** = Healthy life expectancy × Freedom to make life choices - те, що людина за довго проживе здоровою, не зробить її щасливою, якщо вона не буде мати змоги приймати рішення за своє життя
                  
    **Life_support** = Healthy life expectancy × Social support - те, що людина за довго проживе здоровою, не зробить її щасливою, якщо вона не буде мати підтримки рідних та друзів
                  
    **Life_freedom_support** = Healthy life expectancy × Social support x Freedom to make life choice - комплексний фактор кількості здорових років життя, рівня соціальної підтримки та свободи життєвих рішень
                 
    **Delta_Ladder_score** — різниця Ladder score у порівнянні з попереднім роком
                  
    **AI_index_base** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести базових ознак (Healthy life expectancy, Log GDP per capita, Social support, Perceptions of corruption, Freedom to make life choices, Generosity) 
                
        **AI_index_y_region** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести базових ознак (Healthy life expectancy, Log GDP per capita, Social support, Perceptions of corruption, Freedom to make life choices, Generosity), року та регіону 
                
        **AI_index_new_features** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score) 
                
        **AI_index_new_features_y_reg** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score), року та регіону 
                
        **Autoencoder_Index_base** - індекс щастя, побудований за допомогою Sparse-автоенкодера на основі шести базових ознак (Healthy life expectancy, Log GDP per capita, Social support, Perceptions of corruption, Freedom to make life choices, Generosity) 
                
        **Autoencoder_Index_new_features** - індекс щастя, побудований за допомогою Sparse-автоенкодера на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score) 
                
        **VAE_base** - індекс щастя, побудований за допомогою варіаційного автоенкодера (VAE) на основі шести базових ознак (Healthy life expectancy, Log GDP per capita,  Social support, Perceptions of corruption, Freedom to make life choices, Generosity) 
                
        **VAE_new_features** - індекс щастя, побудований за допомогою варіаційного автоенкодера (VAE) на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score)
    """)

# Матриця кореляцій (heatmap)
st.subheader("Матриця кореляцій")

num_cols = [
    "Year", "Ladder score", "Healthy life expectancy", "Log GDP per capita",
    "Social support", "Perceptions of corruption", "Freedom to make life choices",
    "Generosity", "Freedom_trust_index", "Generosity_normalized", "Life_freedom",
    "Life_support", "Life_freedom_support", "Delta_Ladder_score",
    "AI_index_base", "AI_index_y_region", "AI_index_new_features",
    "AI_index_new_features_y_reg", "Autoencoder_Index_base",
    "Autoencoder_Index_new_features", "VAE_base", "VAE_new_features"
]

corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
st.pyplot(fig)

# Top-10 країн за індексом
st.subheader("Топ-10 країн за обраним індексом")

index_options = [
    "Log GDP per capita", "Healthy life expectancy", "Social support",
    "Perceptions of corruption", "Freedom to make life choices", "Generosity",
    "Freedom_trust_index", "Generosity_normalized", "Life_freedom",
    "Life_support", "Life_freedom_support", "Delta_Ladder_score",
    "Ladder score", "AI_index_base", "AI_index_y_region",
    "AI_index_new_features", "AI_index_new_features_y_reg",
    "Autoencoder_Index_base", "Autoencoder_Index_new_features",
    "VAE_base", "VAE_new_features"
]

year = st.slider("Оберіть рік", int(df["Year"].min()), int(df["Year"].max()), 2020)
index_choice = st.selectbox("Оберіть індекс", index_options)
mode = st.radio("Показати:", ["Топ-10 перших", "Топ-10 останніх"])

filtered = df[df["Year"] == year].copy()
filtered = filtered[["Country", index_choice]].dropna()

if mode == "Топ-10 перших":
    top_countries = filtered.nlargest(10, index_choice)
else:
    top_countries = filtered.nsmallest(10, index_choice)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_countries, x=index_choice, y="Country", ax=ax, palette="viridis")
st.pyplot(fig)
