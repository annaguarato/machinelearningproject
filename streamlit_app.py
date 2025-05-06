import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# Funzione per generare dati
def generate_data(n=300):
    np.random.seed(0)
    ore_da_soli = np.random.randint(1, 10, n)
    eventi_sociali = np.random.randint(0, 7, n)
    attivita_individuali = np.random.randint(1, 6, n)
    uso_social = np.random.randint(10, 300, n)

    personalità = []
    for i in range(n):
        score = (
            ore_da_soli[i] * 0.4 +
            (5 - eventi_sociali[i]) * 0.3 +
            attivita_individuali[i] * 0.2 +
            (300 - uso_social[i]) * 0.1 / 300
        )
        personalità.append("Introverso" if score > 4 else "Estroverso")

    return pd.DataFrame({
        'Ore da soli': ore_da_soli,
        'Eventi sociali': eventi_sociali,
        'Attività individuali (1–5)': attivita_individuali,
        'Uso social media (min)': uso_social,
        'Personalità': personalità
    })

# Titolo app
st.title("🌐 Classificazione della Personalità (Introverso vs Estroverso) con KNN")

# Generazione e visualizzazione dati
n = st.slider("Numero di individui simulati", 100, 1000, 300, 50)
data = generate_data(n)
st.subheader("📊 Mappa di correlazione")
fig_corr = plt.figure(figsize=(6, 4))
sns.heatmap(data.drop(columns='Personalità').corr(), annot=True, cmap="coolwarm")
st.pyplot(fig_corr)

# Split dataset
X = data.drop(columns='Personalità')
y = data['Personalità']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Addestramento modello
k = st.slider("🔧 Numero di vicini (k)", 1, 15, 5, 1)
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Risultati modello
st.subheader("📈 Valutazione del Modello")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

# Grafico 2D interattivo
st.subheader("🧭 Distribuzione personalità")
fig2 = px.scatter(data, x="Ore da soli", y="Uso social media (min)", color="Personalità", 
                  hover_data=["Eventi sociali", "Attività individuali (1–5)"])
st.plotly_chart(fig2)

# Input utente per previsione
st.subheader("🧪 Prova tu stesso!")
col1, col2 = st.columns(2)
with col1:
    ore_soli = st.slider("Ore passate da solo al giorno", 1, 12, 5)
    eventi = st.slider("Eventi sociali a settimana", 0, 7, 2)
with col2:
    attivita_ind = st.slider("Preferenza per attività individuali (1–5)", 1, 5, 3)
    uso_social = st.slider("Tempo sui social media (min/giorno)", 0, 300, 120)

# Predizione personalità
input_df = pd.DataFrame({
    'Ore da soli': [ore_soli],
    'Eventi sociali': [eventi],
    'Attività individuali (1–5)': [attivita_ind],
    'Uso social media (min)': [uso_social]
})

prediction = model.predict(input_df)[0]
st.markdown(f"### 🔮 **Risultato: Sei probabilmente un _{prediction}_!**")

# Mostra dati
st.subheader("🧾 Dati simulati")
st.dataframe(data.head(100))
