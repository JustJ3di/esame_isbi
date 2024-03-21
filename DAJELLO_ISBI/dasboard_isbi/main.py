import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


# Funzione per caricare i dati
@st.cache_resource
def load_data():
    df = pd.read_csv("coin_Bitcoin.csv", parse_dates=['Date'])
    return df[['Date', 'Open', 'Close']]

def main():
    st.title("Dashboard di Analisi e Predizione per coin_Bitcoin")
    data = load_data()

    # Sidebar per filtri
    st.sidebar.subheader("Filtri Visualizzazione")
    start_date = st.sidebar.date_input("Data Iniziale", data['Date'].min().date())
    end_date = st.sidebar.date_input("Data Finale", data['Date'].max().date())

    # Filtra i dati in base all'intervallo di date selezionato
    filtered_data = data[(data['Date'].dt.date >= start_date) & (data['Date'].dt.date <= end_date)]

    # Addestramento del modello di regressione
    x = filtered_data[['Open']]
    y = filtered_data['Close']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)

    # Grafico scatterplot con la regressione
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=filtered_data['Open'], y=filtered_data['Close'], ax=ax)
    sns.lineplot(x=x_test['Open'], y=model.predict(x_test), color='red', ax=ax)
    plt.xlabel('Open')
    plt.ylabel('Close')
    plt.title('Regressione Lineare: Open vs Close')
    st.pyplot(fig)

    # Calcolo del coefficiente di determinazione (R^2) e dell'errore quadratico medio (RMSE)
    r2_score = model.score(x_test, y_test)
    rmse = root_mean_squared_error(y_test, model.predict(x_test))

    st.subheader("Valutazione del Modello di Regressione")
    st.write(f"Coefficiente di Determinazione (R^2): {r2_score}")
    st.write(f"Errore Quadratico Medio (RMSE): {rmse}")

    # Standardizzazione dei dati
    standardized_data = (data[['Open', 'Close']] - data[['Open', 'Close']].mean()) / data[['Open', 'Close']].std()

    # Calcolo della matrice di varianza-covarianza
    cov_matrix = standardized_data[['Open', 'Close']].cov()

    # Visualizzazione della matrice di varianza-covarianza come tabella
    st.subheader("Matrice di Varianza-Covarianza")
    st.dataframe(cov_matrix)

    # Sezione di previsione
    st.subheader("Previsione dei Prezzi di Chiusura")
    st.write("Utilizzando il modello di regressione, possiamo fare una previsione dei prezzi di chiusura.")

    # Previsione per un valore di apertura specifico
    open_price = st.slider("Prezzo di Apertura", float(filtered_data['Open'].min()), float(filtered_data['Open'].max()))
    predicted_close_price = model.predict([[open_price]])

    st.write(f"Prezzo di Apertura: {open_price}")
    st.write(f"Prezzo di Chiusura Previsto: {predicted_close_price[0]}")

if __name__ == "__main__":
    main()