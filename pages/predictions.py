import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import CubicSpline

@st.cache_data
def load_data():
    return pd.read_csv("WHR_all2.csv")

df = load_data()

st.title("Прогнозування індексу на основі обраних ознак")

# Налаштування користувача
indices = [
    "AI_index_base", "AI_index_y_region", "AI_index_new_features",
    "AI_index_new_features_y_reg", "Autoencoder_Index_base",
    "Autoencoder_Index_new_features", "VAE_base", "VAE_new_features"
]

target_index = st.selectbox("Оберіть індекс", indices)

all_features = ["Healthy life expectancy", "Log GDP per capita", "Social support", "Perceptions of corruption", 
                "Freedom to make life choices", "Generosity", "Freedom_trust_index", "Generosity_normalized",
                "Life_freedom", "Life_support", "Life_freedom_support", "Delta_Ladder_score", "Year", "Regional_encoded"]

selected_features = st.multiselect("Оберіть ознаки", all_features, default=all_features[:4])

index_model_choice = st.selectbox("Модель прогнозування індексу", ["Random Forest", "XGBoost"])

if st.button("Прогнозувати індекс"):
    st.subheader("Результати прогнозування індексу")
    
    # Підготовка даних
    df_model = df.dropna(subset=selected_features + [target_index])
    X = df_model[selected_features]
    y = df_model[target_index]
    
    st.write(f"Кількість спостережень для навчання: {len(X)}")
    
    # Розділення на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Масштабування ознак
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Вибір та навчання моделі
    if index_model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    
    # Прогноз на тестовій вибірці
    y_pred = model.predict(X_test_scaled)
    
    # Метрики якості
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write("**Метрики якості прогнозу:**")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"R²: {r2:.4f}")
    
    # Діаграма: реальні vs прогнозовані значення (по індексах)
    st.subheader("Реальні vs Прогнозовані значення індексу")

    x_axis = y_test.index  # індекси тестових значень

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(x_axis, y_test, color='green', alpha=0.6, label='Реальні значення')
    ax.scatter(x_axis, y_pred, color='blue', alpha=0.6, label='Прогнозовані значення')

    ax.set_ylabel(target_index)
    ax.set_title('Реальні vs Прогнозовані значення')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    st.pyplot(fig)

st.title("Прогнозування індексу за допомогою екстраполяції")

with st.expander("Детальніше"):
    st.markdown("""
    Всього у датасеті є дані за 21 рік (2005 - 2025). Ця кількість даних є замалою для побудови якісної моделі часових рядів (LSTM, ARIMA тощо). Тому для прогнозування індексу на 2026-2030 роки використовується метод екстраполяції на основі історичних даних (2005-2025) за допомогою:
    - Лінійної регресії (лінійна екстраполяція) - будує пряму між кожною парою сусідніх точок, передбачаючи значення між ними. Для прогнозу за межами відомих точок, вона продовжує цю пряму.
    - Поліноміальної регресії (поліноміальна екстраполяція) - використовує поліном для моделювання залежності між змінними.
    - Кубічного сплайну (сплайн-екстраполяція) - будує гладку криву з кубічних поліномів між точками, забезпечуючи неперервність похідних. Для прогнозу продовжує криву останнього сегмента сплайна. Може вести себе дещо непередбачено за межами відомих даних.
    - Ковзного середнього (moving average) - прогнозує нові значення як середнє останніх кількох спостережень (розмір вікна), згладжуючи коливання.

    Для поліноміальної регресії можна обрати ступінь полінома (2-5), для ковзного середнього - розмір вікна (2-5).
    Метрики якості (MSE, RMSE, MAE, R²) обчислюються на історичних даних (2005-2025), оскільки реальних даних для прогнозу у нас немає.
    """)

# Налаштування користувача 
indices = ["Ladder score",
    "AI_index_base", "AI_index_y_region", "AI_index_new_features",
    "AI_index_new_features_y_reg", "Autoencoder_Index_base",
    "Autoencoder_Index_new_features", "VAE_base", "VAE_new_features"
]

target_index = st.selectbox("Оберіть індекс для прогнозування", indices, key="target_index")
country = st.selectbox("Оберіть країну", df["Country"].unique(), key="country_select")
future_year = st.slider("Виберіть верхню межу прогнозу", 2026, 2030, 2030, key="future_year")
method = st.selectbox("Метод екстраполяції", ["Лінійна", "Поліноміальна", "Кубічний сплайн", "Ковзне середнє"], key="method_select")
poly_degree = st.slider("Ступінь полінома (для поліноміальної регресії)", 2, 5, 3, key="poly_degree")
window_size = st.slider("Розмір вікна (для ковзного середнього)", 2, 5, 3, key="window_size")

if st.button("Прогнозувати"):

    # Підготовка даних
    df_country = df[(df["Country"] == country) & (df["Year"].between(2005, 2025))][["Year", target_index]].dropna()

    if len(df_country) < 5:
        st.warning("Недостатньо даних для прогнозування цієї країни.")
    else:
        X_hist = df_country["Year"].values
        y_hist = df_country[target_index].values

        future_years = np.arange(2026, future_year + 1)

        # Екстраполяція
        if method == "Лінійна":
            model = LinearRegression()
            model.fit(X_hist.reshape(-1,1), y_hist)
            y_pred = model.predict(np.concatenate([X_hist, future_years]).reshape(-1,1))

        elif method == "Поліноміальна":
            model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
            model.fit(X_hist.reshape(-1,1), y_hist)
            y_pred = model.predict(np.concatenate([X_hist, future_years]).reshape(-1,1))

        elif method == "Кубічний сплайн":
            cs = CubicSpline(X_hist, y_hist, bc_type='natural', extrapolate=True)
            y_pred = cs(np.concatenate([X_hist, future_years]))

        elif method == "Ковзне середнє":
            y_pred_hist = y_hist.copy()
            y_pred = list(y_hist)
            for i in range(len(future_years)):
                if len(y_pred) < window_size:
                    mean_val = np.mean(y_pred)
                else:
                    mean_val = np.mean(y_pred[-window_size:])
                y_pred.append(mean_val)
            y_pred = np.array(y_pred)

        # Розділяємо на історичні та майбутні значення
        y_pred_hist = y_pred[:len(X_hist)]
        y_pred_future = y_pred[len(X_hist):]

        # Обмежуємо прогноз в межах 0-10
        y_pred = np.clip(y_pred, 0, 10)
        y_pred_hist = y_pred[:len(X_hist)]
        y_pred_future = y_pred[len(X_hist):]

        # Метрики якості на історичних даних 
        mse = mean_squared_error(y_hist, y_pred_hist)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_hist, y_pred_hist)
        r2 = r2_score(y_hist, y_pred_hist)

        st.subheader("Метрики на історичних даних (2005-2025)")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"R²: {r2:.4f}")

        # Графік прогнозу
        plt.figure(figsize=(12,6))
        plt.plot(X_hist, y_hist, marker="o", label="Історичні дані")
        plt.plot(np.concatenate([X_hist, future_years]), y_pred, marker="x", linestyle="--", color="red", label="Прогноз")
        plt.axvline(x=2025, color='gray', linestyle=':', label='Початок прогнозу (2026 - 2030)')
        plt.xlabel("Рік")
        plt.ylabel(target_index)
        plt.title(f"Прогноз '{target_index}' для {country} ({method})")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

        # Таблиця прогнозу
        df_forecast = pd.DataFrame({
            "Year": future_years,
            "Forecast": y_pred_future
        })
        st.subheader("Прогнозовані значення")
        st.dataframe(df_forecast)