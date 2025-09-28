import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import shap
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Завантаження даних
df = pd.read_csv("./WHR_all2.csv")

# Функція для SHAP-аналізу 
def analyze_index_shap(target_variable, features_to_use, model_type):    
    # Підготовка даних
    X = df[features_to_use].copy()
    y = df[target_variable].copy()
    
    # Видаляємо пропущені значення
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        return None, None, None, None
    
    # Розділення на тренувальну та тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
    elif model_type == "XGBoost":
        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    
    return model, shap_values, X_test, features_to_use

index_weights = {
    "AI_index_base": {
        "Log GDP per capita": 0.478966,
        "Social support": 0.149858,
        "Healthy life expectancy": 0.149445,
        "Freedom to make life choices": 0.122940,
        "Generosity": 0.051405,
        "Perceptions of corruption": 0.047386
    },
    "AI_index_y_region": {
        "Log GDP per capita": 0.479656,
        "Regional_encoded": 0.164255,
        "Social support": 0.097470,
        "Healthy life expectancy": 0.064584,
        "Freedom to make life choices": 0.088336,
        "Generosity": 0.039057,
        "Perceptions of corruption": 0.039736,
        "Year": 0.026906
    },
    "AI_index_new_features": {
        "Life_support": 0.227793,
        "Life_freedom_support": 0.402605,
        "Life_freedom": 0.156697,
        "Freedom_trust_index": 0.105645,
        "Delta_Ladder_score": 0.069140,
        "Generosity_normalized": 0.038120
    },
    "AI_index_new_features_y_reg": {
        "Life_support": 0.115439,
        "Life_freedom_support": 0.407371,
        "Life_freedom": 0.047260,
        "Freedom_trust_index": 0.086775,
        "Delta_Ladder_score": 0.063564,
        "Generosity_normalized": 0.028430,
        "Regional_encoded": 0.219798,
        "Year": 0.031363
    }
}

st.title("Аналіз індексів щастя")

# Список усіх індексів
index_columns = [
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

# Порівняння індексів
st.header("Порівняння індексів")

with st.expander("Пояснення до індексів"):
    st.markdown("""
        **AI_index_base** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести базових ознак (Healthy life expectancy, Log GDP per capita, Social support, Perceptions of corruption, Freedom to make life choices, Generosity) 
                
        **AI_index_y_region** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести базових ознак (Healthy life expectancy, Log GDP per capita, Social support, Perceptions of corruption, Freedom to make life choices, Generosity), року та регіону 
                
        **AI_index_new_features** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score) 
                
        **AI_index_new_features_y_reg** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score), року та регіону 
                
        **Autoencoder_Index_base** - індекс щастя, побудований за допомогою Sparse-автоенкодера на основі шести базових ознак (Healthy life expectancy, Log GDP per capita, Social support, Perceptions of corruption, Freedom to make life choices, Generosity) 
                
        **Autoencoder_Index_new_features** - індекс щастя, побудований за допомогою Sparse-автоенкодера на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score) 
                
        **VAE_base** - індекс щастя, побудований за допомогою варіаційного автоенкодера (VAE) на основі шести базових ознак (Healthy life expectancy, Log GDP per capita,  Social support, Perceptions of corruption, Freedom to make life choices, Generosity) 
                
        **VAE_new_features** - індекс щастя, побудований за допомогою варіаційного автоенкодера (VAE) на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score)
    """)

selected_indices = st.multiselect(
    "Оберіть індекси для порівняння:", 
    index_columns,
    default=["Ladder score", "AI_index_base"]
)

selected_country_compare = st.selectbox(
    "Оберіть країну для порівняння:",
    df["Country"].unique(),
    key="compare_country"
)

compare_data = df[df["Country"] == selected_country_compare]

fig2, ax2 = plt.subplots(figsize=(12, 6))

for idx in selected_indices:
    ax2.plot(
        compare_data["Year"], 
        compare_data[idx], 
        marker="o", 
        linestyle="-", 
        label=idx
    )

ax2.set_title(f"Порівняння індексів у {selected_country_compare} (2005–2025)")
ax2.set_xlabel("Рік")
ax2.set_ylabel("Значення")
ax2.legend()
st.pyplot(fig2)

st.header("Важливість ознак для базових індексів")

with st.expander("Пояснення до індексів"):
    st.markdown("""
        **AI_index_base** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести базових ознак (Healthy life expectancy, Log GDP per capita, Social support, Perceptions of corruption, Freedom to make life choices, Generosity) 
                
        **AI_index_y_region** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести базових ознак (Healthy life expectancy, Log GDP per capita, Social support, Perceptions of corruption, Freedom to make life choices, Generosity), року та регіону 
                
        **AI_index_new_features** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score) 
                
        **AI_index_new_features_y_reg** - індекс щастя, побудований "вручну" за допомогою SHAP-аналізу на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score), року та регіону    
    """)

selected_index_weights = st.selectbox(
    "Оберіть індекс для перегляду ваг:", 
    list(index_weights.keys())
)

with st.expander("Пояснення принципу ваг"):
    st.markdown("""Вага - це показник важливості кожної ознаки для прогнозування індексу. Чим вища вага, тим більший вплив має ознака на модель.
                Для індексів, побудованих "вручну" (AI_index_*), ваги отримані шляхом аналізу SHAP-важливості для моделей Random Forest та XGBoost на оригінальних даних Ladder score.
                Ці індекси були створені як лінійна комбінація ознак з вагами, що відображають їхній внесок у прогнозування щастя.
                Вони були створені з метою забезпечення прозорості та інтерпретованості, на відміну від складних моделей машинного навчання, які можуть бути менш зрозумілими.
                Практична користь цих індексів полягає в тому, що вони дозволяють легко оцінити вплив кожної ознаки на загальний індекс щастя та подивитися, чи може Ladder score бути пояснений як лінійна комбінація цих ознак.
                Також ваги були нормалізовані так, щоб їх сума дорівнювала 1, що дозволяє легко порівнювати відносну важливість ознак.
                """)

if selected_index_weights in index_weights:
    weights = index_weights[selected_index_weights]
    
    # Створюємо DataFrame для відображення
    weights_df = pd.DataFrame({
        'Ознака': list(weights.keys()),
        'Вага': list(weights.values())
    }).sort_values('Вага', ascending=False)
    
    st.subheader(f"Ваги для {selected_index_weights}")
    st.dataframe(weights_df)
    
    # Візуалізація ваг
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(weights_df))
    ax.barh(y_pos, weights_df['Вага'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(weights_df['Ознака'])
    ax.set_xlabel('Вага')
    ax.set_title(f'Розподіл ваг для {selected_index_weights}')
    ax.invert_yaxis()
    st.pyplot(fig)


# Інтерактивний SHAP-аналіз
st.header("SHAP-аналіз базових індексів")

# Визначення ознак для кожного індексу
index_features_definitions = {
    "AI_index_base": [
        "Log GDP per capita", "Social support", "Healthy life expectancy",
        "Freedom to make life choices", "Generosity", "Perceptions of corruption"
    ],
    "AI_index_y_region": [
        "Log GDP per capita", "Social support", "Healthy life expectancy",
        "Freedom to make life choices", "Generosity", "Perceptions of corruption",
        "Regional_encoded", "Year"
    ],
    "AI_index_new_features": [
        "Life_support", "Life_freedom_support", "Life_freedom",
        "Freedom_trust_index", "Delta_Ladder_score", "Generosity_normalized"
    ],
    "AI_index_new_features_y_reg": [
        "Life_support", "Life_freedom_support", "Life_freedom",
        "Freedom_trust_index", "Delta_Ladder_score", "Generosity_normalized",
        "Regional_encoded", "Year"
    ]
}

selected_index_shap = st.selectbox(
    "Оберіть індекс для SHAP-аналізу:",
    list(index_weights.keys())
)

model_option_shap = st.selectbox(
    "Оберіть модель для SHAP-аналізу:",
    ["Random Forest", "XGBoost"],
    key="shap_model"
)

if st.button("Запустити SHAP-аналіз індексу"):
    if selected_index_shap in index_features_definitions:
        features = index_features_definitions[selected_index_shap]
        
        with st.spinner("Виконується SHAP-аналіз..."):
            model, shap_values, X_test, features_used = analyze_index_shap(
                selected_index_shap, features, model_option_shap
            )
            
            if model is not None:
                # Порівняння: ваші ваги vs SHAP-важливість
                st.subheader("Порівняння: Початкові ваги vs SHAP-важливість")
                
                # Ваші ваги
                init_weights = index_weights[selected_index_shap]
                
                # SHAP-важливість
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                shap_importance = mean_abs_shap / mean_abs_shap.sum()
                
                # Створюємо DataFrame для порівняння
                comparison_data = []
                for i, feature in enumerate(features_used):
                    init_weight = init_weights.get(feature, 0)
                    shap_weight = shap_importance[i] if i < len(shap_importance) else 0
                    comparison_data.append({
                        'Ознака': feature,
                        'Початкові ваги': init_weight,
                        'SHAP-важливість': shap_weight,
                        'Різниця': abs(init_weight - shap_weight)
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Візуалізація порівняння
                fig_comp, ax_comp = plt.subplots(figsize=(12, 8))
                x_pos = np.arange(len(comparison_df))
                width = 0.35

                ax_comp.bar(x_pos - width/2, comparison_df['Початкові ваги'], width, label='Початкові ваги')
                ax_comp.bar(x_pos + width/2, comparison_df['SHAP-важливість'], width, label='SHAP-важливість')
                
                ax_comp.set_xlabel('Ознаки')
                ax_comp.set_ylabel('Ваги')
                ax_comp.set_title('Порівняння ваг')
                ax_comp.set_xticks(x_pos)
                ax_comp.set_xticklabels(comparison_df['Ознака'], rotation=45)
                ax_comp.legend()
                plt.tight_layout()
                st.pyplot(fig_comp)

                # Summary plot
                st.subheader("Summary plot")
                fig_summary, ax_summary = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
                st.pyplot(fig_summary)

                st.dataframe(comparison_df)
                
                # Пояснення розбіжностей
                st.subheader("Пояснення розбіжностей")
                st.markdown("""
                **Чому ваги відрізняються:**
                
                1. **Різні цільові змінні**: Початкові ваги отримані для прогнозування 'Ladder score', 
                   а SHAP-аналіз робиться для прогнозування конкретного індексу

                2. **Різні моделі**: Початкові ваги базуються на середньому SHAP від Random Forest та XGBoost
                   для оригінальних даних
                
                3. **Лінійність vs нелінійність**: Створений індекс - лінійна комбінація, а ML-моделі 
                   виявляють нелінійні залежності
                """)

st.header("SHAP-аналіз індексів, отриманих за допомогою автоенкодера")

with st.expander("Пояснення до індексів"):
    st.markdown("""                
        **Autoencoder_Index_base** - індекс щастя, побудований за допомогою Sparse-автоенкодера на основі шести базових ознак (Healthy life expectancy, Log GDP per capita, Social support, Perceptions of corruption, Freedom to make life choices, Generosity) 
                
        **Autoencoder_Index_new_features** - індекс щастя, побудований за допомогою Sparse-автоенкодера на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score) 
                
        **VAE_base** - індекс щастя, побудований за допомогою варіаційного автоенкодера (VAE) на основі шести базових ознак (Healthy life expectancy, Log GDP per capita,  Social support, Perceptions of corruption, Freedom to make life choices, Generosity) 
                
        **VAE_new_features** - індекс щастя, побудований за допомогою варіаційного автоенкодера (VAE) на основі шести штучно створених ознак (Freedom to make life choices, Generosity, Freedom_trust_index, Generosity_normalized, Life_freedom, Life_support, Life_freedom_support, Delta_Ladder_score)
    """)

autoencoder_index_cols = [
    "Autoencoder_Index_base",
    "Autoencoder_Index_new_features",
    "VAE_base",
    "VAE_new_features"
]

selected_index_autoencoder = st.selectbox(
    "Оберіть індекс для SHAP-аналізу:", 
    autoencoder_index_cols
)

with st.expander("Пояснення принципу побудови індексів"):
    st.markdown("""Індекси, побудовані за допомогою автоенкодера, є результатом моделей машинного навчання, які автоматично вивчають представлення даних.
                На відміну від індексів, створених "вручну" з використанням SHAP-аналізу, ці індекси не мають явних ваг для кожної ознаки. Це дає їм змогу знаходити складні, нелінійні залежності між ознаками.
                У роботі використовувалися два типи автоенкодерів: Sparse-автоенкодер та варіаційний автоенкодер (VAE).
                Sparse-автоенкодер навчається відтворювати вхідні дані, при цьому накладаючи обмеження на кількість активних нейронів у прихованому шарі, що сприяє виявленню найважливіших ознак.
                Варіаційний автоенкодер (VAE) навчається не лише відтворювати вхідні дані, але й моделювати їх розподіл, що дозволяє генерувати нові зразки та краще захоплювати структуру даних.
                Практична користь цих індексів полягає в тому, що вони можуть виявляти складні патерни у даних, які можуть бути пропущені лінійними моделями.
                Однак, через відсутність явних ваг, інтерпретація цих індексів може бути складнішою.
                """)

# Визначення рекомендованих ознак для кожного індексу
recommended_features = {
    "Autoencoder_Index_base": [
        "Healthy life expectancy", "Log GDP per capita", "Social support", 
        "Perceptions of corruption", "Freedom to make life choices", "Generosity"
    ],
    "VAE_base": [
        "Healthy life expectancy", "Log GDP per capita", "Social support", 
        "Perceptions of corruption", "Freedom to make life choices", "Generosity"
    ],
    "Autoencoder_Index_new_features": [
        "Life_freedom", "Life_support", "Life_freedom_support", 
        "Freedom_trust_index", "Delta_Ladder_score", "Generosity_normalized"
    ],
    "VAE_new_features": [
        "Life_freedom", "Life_support", "Life_freedom_support", 
        "Freedom_trust_index", "Delta_Ladder_score", "Generosity_normalized"
    ]
}

features_list = [
    "Healthy life expectancy", "Log GDP per capita", "Social support", "Perceptions of corruption",
    "Freedom to make life choices", "Generosity", "Freedom_trust_index", "Generosity_normalized",
    "Life_freedom", "Life_support", "Life_freedom_support", "Delta_Ladder_score"
]

# Отримуємо рекомендовані ознаки для обраного індексу
default_features = recommended_features.get(selected_index_autoencoder, features_list[:6])

autoencoder_features = st.multiselect(
    "Оберіть ознаки для SHAP-аналізу:", 
    features_list,
    default=default_features
)

model_option_shap_autoencoder = st.selectbox(
    "Оберіть модель для SHAP-аналізу:",
    ["Random Forest", "XGBoost"],
    key="shap_model_autoencoder"
)

# Додаємо опцію для вибору типу візуалізації
plot_type = st.radio(
    "Оберіть тип SHAP-візуалізації:",
    ["Bar plot", "Summary plot", "Обидва"],
    horizontal=True
)

if st.button("Запустити SHAP-аналіз автоенкодера"):
    if selected_index_autoencoder in autoencoder_index_cols and len(autoencoder_features) > 3:
        with st.spinner("Виконується SHAP-аналіз..."):
            try:
                # Підготовка даних
                X = df[autoencoder_features].copy()
                y = df[selected_index_autoencoder].copy()
                
                # Видаляємо пропущені значення
                mask = X.notna().all(axis=1) & y.notna()
                X = X[mask]
                y = y[mask]
                
                if len(X) == 0:
                    st.error("Недостатньо даних для аналізу після видалення пропущених значень.")
                else:
                    # Навчання моделі та отримання SHAP значень
                    from sklearn.model_selection import train_test_split
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    if model_option_shap_autoencoder == "Random Forest":
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)
                        
                    elif model_option_shap_autoencoder == "XGBoost":
                        import xgboost as xgb
                        model = xgb.XGBRegressor(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)
                    
                    # Отримання важливості ознак
                    if len(shap_values.shape) > 2:
                        shap_values = shap_values[0]
                    
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    shap_importance = mean_abs_shap / mean_abs_shap.sum()
                    
                    # Створюємо DataFrame з результатами
                    importance_df = pd.DataFrame({
                        'Ознака': autoencoder_features,
                        'SHAP-важливість': shap_importance,
                        'Абсолютна важливість': mean_abs_shap
                    }).sort_values('SHAP-важливість', ascending=False)
                    
                    st.subheader(f"Результати SHAP-аналізу для {selected_index_autoencoder}")
                    st.write(f"**Модель:** {model_option_shap_autoencoder}")
                    st.write(f"**Кількість спостережень:** {len(X_test)}")
                    
                    # Bar plot важливості ознак
                    if plot_type in ["Bar plot", "Обидва"]:
                        st.subheader("Bar plot важливості ознак")
                        
                        fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
                        y_pos = np.arange(len(importance_df))
                        
                        bars = ax_bar.barh(y_pos, importance_df['SHAP-важливість'])
                        ax_bar.set_yticks(y_pos)
                        ax_bar.set_yticklabels(importance_df['Ознака'])
                        ax_bar.set_xlabel('Відносна важливість (SHAP)')
                        ax_bar.set_title(f'Важливість ознак для {selected_index_autoencoder}\n({model_option_shap_autoencoder})')
                        ax_bar.invert_yaxis()  # Найважливіші ознаки зверху
                        
                        # Додаємо значення на стовпці
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax_bar.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                      f'{width:.3f}', ha='left', va='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig_bar)
                    
                    # Таблиця з результатами
                    st.subheader("Таблиця важливості ознак")
                    
                    display_df = importance_df.copy()
                    display_df['SHAP-важливість'] = display_df['SHAP-важливість'].round(4)
                    display_df['Абсолютна важливість'] = display_df['Абсолютна важливість'].round(4)
                    
                    st.dataframe(display_df[['Ознака', 'SHAP-важливість', 'Абсолютна важливість']])
                    
                    # Summary plot
                    if plot_type in ["Summary plot", "Обидва"]:
                        st.subheader("Summary plot")
                        fig_summary, ax_summary = plt.subplots(figsize=(10, 8))
                        shap.summary_plot(shap_values, X_test, feature_names=autoencoder_features, show=False)
                        plt.tight_layout()
                        st.pyplot(fig_summary)
                    
                    # Аналіз якості моделі
                    st.subheader("Якість моделі")
                    from sklearn.metrics import r2_score, mean_squared_error
                    
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{r2:.4f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.4f}")
                    # Scatter plot прогнозів vs фактичних значень
                    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))

                    # Фактичні значення
                    ax_scatter.scatter(range(len(y_test)), y_test, alpha=0.7, color='blue', 
                  label='Фактичні значення', s=50)

                    # Прогнозовані значення
                    ax_scatter.scatter(range(len(y_pred)), y_pred, alpha=0.7, color='red', 
                  label='Прогнозовані значення', s=50)

                    ax_scatter.set_xlabel('Індекс спостереження')
                    ax_scatter.set_ylabel('Значення індексу')
                    ax_scatter.set_title(f'Порівняння фактичних та прогнозованих значень (R² = {r2:.3f})')
                    ax_scatter.legend()
                    ax_scatter.grid(True, alpha=0.3)

                    st.pyplot(fig_scatter)
                    
            except Exception as e:
                st.error(f"Помилка під час виконання аналізу: {str(e)}")
                st.info("Перевірте, чи всі обрані ознаки присутні у датафреймі та не містять пропущених значень.")
    
    else:
        st.warning("Будь ласка, оберіть індекс та хоча б три ознаки для аналізу.")
