import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# Define the updated bond ETFs
bond_etfs = {
    "US Bonds (iShares iBoxx $ Investment Grade Corporate Bond)": "LQD",
    "Eurozone Bonds (iShares Euro Corporate Bond)": "IBCX.DE",
    "UK Bonds (iShares £ Corporate Bond)": "SLXX.L",
    "Switzerland Bonds (iShares CHF Corporate Bond)": "CHCORP.SW",
    "Japan Bonds (iShares JP Morgan USD Emerging Markets Corporate Bond)": "EMB",
    "Australia Bonds (iShares Core Corporate Bond ETF)": "ICOR.AX",
    "Singapore Bonds (Nikko AM SGD Investment Grade Corporate Bond ETF)": "MBH.SI"
}

# Define the updated stock indices
stock_indices = {
    "US (S&P 500)": "^GSPC",
    "Japan (Nikkei 225)": "^N225",
    "Singapore (STI)": "^STI",
    "Switzerland (SMI)": "^SSMI",
    "Germany (DAX)": "^GDAXI",
    "Australia (ASX 200)": "^AXJO",
    "China (SSE Composite)": "000001.SS"
}

# Define the updated REIT ETFs
reit_etfs = {
    "US REITs (Vanguard Real Estate ETF)": "VNQ",
    "Japan REITs (iShares MSCI Japan ETF)": "EWJ",
    "Singapore REITs (Lion-Phillip S-REIT ETF)": "CLR.SI",
    "Switzerland REITs (iShares Developed Markets Property Yield CHF Hedged ETF)": "SRECHA.SW",
    "Australia REITs (Vanguard Australian Property ETF)": "VAP.AX",
    "China REITs (Global X MSCI China Real Estate ETF)": "CHIR"
}

# Combine all ETFs into a single dictionary
all_etfs = {**stock_indices, **bond_etfs, **reit_etfs}

# Fetch historical data using yfinance
def fetch_data(etf_ticker, start_date="2015-01-01", end_date="2023-01-01"):
    data = yf.download(etf_ticker, start=start_date, end=end_date)
    return data['Adj Close']

# Streamlit app layout
st.set_page_config(page_title="Portfolio Optimization with Genetic Algorithm", layout="wide")
st.title('AI-Powered Portfolio Optimization')

st.markdown("""
    **Select ETFs from various asset classes and countries** to create an initial portfolio. 
    Then, let the AI optimize and rebalance the portfolio monthly to maximize returns and manage risk.
""")

# Selection of ETFs
st.sidebar.header("Initial Portfolio Selection")
stock_choice = st.sidebar.selectbox("Choose an initial Stock Index ETF", list(stock_indices.keys()))
bond_choice = st.sidebar.selectbox("Choose an initial Bond ETF", list(bond_etfs.keys()))
reit_choice = st.sidebar.selectbox("Choose an initial REIT ETF", list(reit_etfs.keys()))

st.sidebar.markdown(f"### Your initial choices:")
st.sidebar.write(f"**Stock Index**: {stock_indices[stock_choice]} from {stock_choice}")
st.sidebar.write(f"**Bond**: {bond_etfs[bond_choice]} from {bond_choice}")
st.sidebar.write(f"**REIT**: {reit_etfs[reit_choice]} from {reit_choice}")

# User sets the initial weights for their portfolio
stock_weight = st.sidebar.slider("Initial Weight for Stock Index", 0.0, 1.0, 0.5)
bond_weight = st.sidebar.slider("Initial Weight for Bond", 0.0, 1.0, 0.3)
reit_weight = st.sidebar.slider("Initial Weight for REIT", 0.0, 1.0, 0.2)

# Ensure that the weights sum to 1
total_weight = stock_weight + bond_weight + reit_weight
if total_weight != 1:
    st.error("The total weight must sum to 1. Please adjust your weights.")
else:
    # Fetch data for all ETFs (not just the user's initial selection)
    etf_data = {name: fetch_data(ticker) for name, ticker in all_etfs.items()}
    
    # Combine the data into a single DataFrame
    portfolio_df = pd.DataFrame({
        name: etf for name, etf in etf_data.items()
    }).dropna()

    # Monthly returns for all ETFs
    monthly_returns = portfolio_df.pct_change().dropna()

    # Initial portfolio performance (user's starting allocation)
    initial_portfolio = np.cumprod(1 + (monthly_returns[stock_choice] * stock_weight +
                                        monthly_returns[bond_choice] * bond_weight +
                                        monthly_returns[reit_choice] * reit_weight))

    # Set up the Genetic Algorithm
    def evaluate_portfolio(individual):
        # individual is a list of 3 ETF tickers and corresponding weights
        etf_weights = np.array([individual[3], individual[4], individual[5]])

        # Retrieve return series for the chosen ETFs
        stock_returns = monthly_returns[individual[0]]
        bond_returns = monthly_returns[individual[1]]
        reit_returns = monthly_returns[individual[2]]

        # Calculate portfolio return and risk
        portfolio_return = np.mean(
            stock_returns * etf_weights[0] +
            bond_returns * etf_weights[1] +
            reit_returns * etf_weights[2]
        )
        portfolio_risk = np.std(
            stock_returns * etf_weights[0] +
            bond_returns * etf_weights[1] +
            reit_returns * etf_weights[2]
        )

        sharpe_ratio = portfolio_return / portfolio_risk
        return sharpe_ratio,

    # Custom crossover function: perform random selection for ETF names, blend for weights
    def custom_crossover(ind1, ind2, alpha=0.5):
        # Perform random selection for ETF names
        for i in range(3):  # The first three elements are ETF names
            if random.random() > 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        # Perform cxBlend for the weights (last 3 elements)
        for i in range(3, 6):
            x1 = ind1[i]
            x2 = ind2[i]
            gamma = (1. - alpha) * random.random() + alpha
            ind1[i] = (1. - gamma) * x1 + gamma * x2
            ind2[i] = gamma * x1 + (1. - gamma) * x2

        return ind1, ind2

    # Custom mutation function
    def custom_mutation(individual, mu=0, sigma=0.2, indpb=0.2):
        # Mutate ETF names by random selection
        for i in range(3):  # The first three elements are ETF names
            if random.random() < indpb:
                individual[i] = random.choice(list(all_etfs.keys()))

        # Mutate weights using Gaussian mutation
        for i in range(3, 6):  # The last three elements are weights
            if random.random() < indpb:
                individual[i] += random.gauss(mu, sigma)
                individual[i] = max(0, min(1, individual[i]))  # Keep weights within [0, 1]

        return individual,

    # Create Genetic Algorithm components
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_etf", np.random.choice, list(all_etfs.keys()))
    toolbox.register("attr_float", np.random.uniform, 0, 1)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_etf, toolbox.attr_etf, toolbox.attr_etf, toolbox.attr_float, toolbox.attr_float, toolbox.attr_float), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_portfolio)
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize the population and run the algorithm
    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the algorithm for 50 generations
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50,
                                              stats=stats, halloffame=hof, verbose=False)

    # Get the optimized weights from the genetic algorithm
    optimized_individual = hof[0]
    optimized_weights = np.array([optimized_individual[3], optimized_individual[4], optimized_individual[5]])

    # AI-optimized portfolio performance
    ai_portfolio = np.cumprod(1 + (monthly_returns[optimized_individual[0]] * optimized_weights[0] +
                                   monthly_returns[optimized_individual[1]] * optimized_weights[1] +
                                   monthly_returns[optimized_individual[2]] * optimized_weights[2]))

    # Plot portfolio performance
    def plot_performance(initial_portfolio, ai_portfolio):
        plt.figure(figsize=(10, 6))
        plt.plot(initial_portfolio, label='User Initial Portfolio', linestyle='--', color='blue', linewidth=2)
        plt.plot(ai_portfolio, label='AI-Optimized Portfolio', color='green', linewidth=2)
        plt.title('Portfolio Performance: User vs AI-Optimized', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Portfolio Value', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        plt.legend(fontsize=12)
        st.pyplot(plt)

    # Display the results
    st.header("Performance Results")
    plot_performance(initial_portfolio, ai_portfolio)

    st.subheader("Optimized Portfolio Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Stock ETF", optimized_individual[0], f"Weight: {optimized_weights[0]:.2f}")
    col2.metric("Bond ETF", optimized_individual[1], f"Weight: {optimized_weights[1]:.2f}")
    col3.metric("REIT ETF", optimized_individual[2], f"Weight: {optimized_weights[2]:.2f}")

    # Count how many times each country/ETF is picked
    country_count = {country: 0 for country in all_etfs.keys()}
    singapore_count = 0

    # Track how often each country is selected
    for ind in population:
        for etf in [ind[0], ind[1], ind[2]]:
            country_count[etf] += 1
            if "Singapore" in etf:
                singapore_count += 1

    # Display country selection statistics
    st.subheader("Country/ETF Selection Frequency")
    country_df = pd.DataFrame.from_dict(country_count, orient='index', columns=['Selection Count'])
    st.bar_chart(country_df)

    # Display Singapore statistics
    st.write(f"Singapore assets were selected {singapore_count} times during rebalancing.")

