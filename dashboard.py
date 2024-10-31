import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Set page config and custom theme
st.set_page_config(
    page_title="Dashboard Pelatihan Dicoding Salman Fadhilurrohman",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.dicoding.com/academies/555',
        'Report a bug': "https://github.com/yourusername/yourrepository/issues",
        'About': "# This is a dashboard created for Dicoding course project."
    }
)

# Custom theme using CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(
            45deg,
            #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #8f00ff
        );
        background-size: 400% 400%;
        animation: rainbow 15s ease infinite;
        color: black; /* Untuk memastikan teks tetap terbaca */
    }
    h2{
        color:black;
    }
    h3{
        color:black;
    }
    .st-emotion-cache-q49buc{
        color:black;
    }
    .st-emotion-cache-efbu8t{
        color:black;
    }
    @keyframes rainbow {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    .stButton>button {
        color: #4F8BF9;
        border-radius: 50%;
        height: 3em;
        width: 3em;
    }
    .stTextInput>div>div>input {
        color: #4F8BF9;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(
            45deg,
            #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #8f00ff
        );
        background-size: 400% 400%;
        animation: rainbow 15s ease infinite;
    }

    @keyframes rainbow {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Set the style and color palette
sns.set_style("whitegrid")
sns.set_palette("deep")

# sns.set(style='whitegrid')

def create_penjualan_df(df):
    penjualan_df = df.resample(rule='D', on='order_date').agg({
        "order_id": "nunique",
        "total_price": "sum"
    }).reset_index()
    
    penjualan_df.rename(columns={
        "order_id": "order_count",
        "total_price": "revenue"
    }, inplace=True)
    
    return penjualan_df

def create_sum_penjualan_barang_df(df):
    sum_penjualan_barang_df = df.groupby("product_name").quantity_x.sum().sort_values(ascending=False).reset_index()
    return sum_penjualan_barang_df

def create_byjk_df(df):
    byjk_df = df.groupby(by="gender").customer_id.nunique().reset_index()
    byjk_df.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    
    return byjk_df

def create_byumur_df(df):
    byumur_df = df.groupby(by="age_group").customer_id.nunique().reset_index()
    byumur_df.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    
    return byumur_df

def create_bydaerah_df(df):
    bydaerah_df = df.groupby(by="state").customer_id.nunique().reset_index()
    bydaerah_df.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    
    return bydaerah_df

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_date": "max",
        "order_id": "nunique",
        "total_price": "sum"
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_date"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    return rfm_df

def create_city_mapping(df):
    city_mapping = df.groupby('city').customer_id.nunique().reset_index()
    city_mapping.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    
    return city_mapping

# Load data
all_df = pd.read_csv("D:/Dataanalyst/python/baru/latihan/project/clean_data.csv")

datetime_columns = ["order_date", "delivery_date"]
all_df.sort_values(by="order_date", inplace=True)
all_df.reset_index(drop=True, inplace=True)

# Convert date columns to datetime format
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

# Get minimum and maximum dates from the dataset
min_date = all_df["order_date"].min()
max_date = all_df["order_date"].max()

# Sidebar setup for Streamlit
with st.sidebar:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    
    start_date, end_date = st.date_input(
        label='Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Filter the dataframe based on the selected date range
main_df = all_df[(all_df["order_date"] >= pd.to_datetime(start_date)) &
                 (all_df["order_date"] <= pd.to_datetime(end_date))]

# Generate summary dataframes
penjualan_harian_df = create_penjualan_df(main_df)  
sum_penjualan_barang_df = create_sum_penjualan_barang_df(main_df)
byjk_df = create_byjk_df(main_df)
byumur_df = create_byumur_df(main_df)
bydaerah_df = create_bydaerah_df(main_df)
rfm_df = create_rfm_df(main_df)

st.header('Dashboard Pelatihan Dicoding Salman Fadhilurrohman ﾟ ⋆ ﾟ ☂︎ ⋆ ﾟﾟ ⋆ ')
# st.subheader('Peramalan 30 hari Kedepan')

# Feature Engineering
penjualan_harian_df['day_of_week'] = penjualan_harian_df['order_date'].dt.dayofweek
penjualan_harian_df['month'] = penjualan_harian_df['order_date'].dt.month
penjualan_harian_df['day'] = penjualan_harian_df['order_date'].dt.day

# Menentukan fitur dan target
features = ['day_of_week', 'month', 'day']
target = 'order_count'
X = penjualan_harian_df[features]
y = penjualan_harian_df[target]

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Membuat DMatrix untuk XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Mengatur parameter XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 5,
    'eta': 0.1
}

# Melatih model
model = xgb.train(params, dtrain, num_boost_round=100)

# Melakukan prediksi
y_pred = model.predict(dtest)

# Evaluasi Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)



# Menentukan jumlah hari ke depan untuk peramalan
forecast_days = 30

# Membuat dataframe untuk tanggal masa depan
last_date = penjualan_harian_df['order_date'].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)

# Membuat dataframe fitur untuk tanggal masa depan
future_df = pd.DataFrame({
    'order_date': future_dates,
    'day_of_week': future_dates.dayofweek,
    'month': future_dates.month,
    'day': future_dates.day
})
future_features = future_df[features]
future_dmatrix = xgb.DMatrix(future_features)

# Melakukan prediksi untuk masa depan
future_pred = model.predict(future_dmatrix)
future_df['predicted_order_count'] = future_pred

# Menampilkan Grafik Peramalan
st.subheader('Peramalan Penjualan 30 Hari ke Depan')
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(penjualan_harian_df['order_date'], penjualan_harian_df['order_count'], label='Penjualan Aktual', color='blue')
ax.plot(future_df['order_date'], future_df['predicted_order_count'], label='Prediksi Penjualan', color='red', linestyle='--')
ax.set_title("Peramalan Penjualan Menggunakan XGBoost", fontsize=20)
ax.set_xlabel("Tanggal", fontsize=15)
ax.set_ylabel("Jumlah Penjualan", fontsize=15)
ax.legend(fontsize=12)
ax.grid(True)
st.pyplot(fig)
# Menampilkan hasil evaluasi di Streamlit
st.write(f"### Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"### Mean Squared Error (MSE): {mse:.2f}")
st.write(f"### Root Mean Squared Error (RMSE): {rmse:.2f}")

# Kesimpulan
st.write("Dari hasil forecasting menggunakan XGBoost, jika dilihat dari hasil evaluasi menggunakan Mean Absolute Error (MAE), Mean Squared Error (MSE), dan Root Mean Squared Error (RMSE), terlihat akurasi dari hasil forecasting cukup baik.")
st.write(f"Dimana MAE merupakan rata-rata kesalahan antara nilai prediksi dan nilai aktual, dengan nilai MAE sebesar {mae:.2f}.")
st.write(f"Kemudian MSE merupakan rata-rata dari kesalahan kuadrat, di mana nilainya cukup kecil, berada di angka {mse:.2f}.")
st.write(f"Terakhir, RMSE (rata-rata deviasi dari prediksi) memiliki nilai sebesar {rmse:.2f}, yang menggambarkan rata-rata deviasi dari hasil prediksi sebesar {rmse:.2f} unit. Ini memberikan gambaran tentang seberapa jauh prediksi melenceng dari nilai aktual.")




# # Menampilkan Tabel Peramalan
# st.subheader('Tabel Peramalan Penjualan 30 Hari ke Depan')
# st.write(future_df[['order_date', 'predicted_order_count']].rename(columns={'order_date': 'Tanggal', 'predicted_order_count': 'Prediksi Penjualan'}))

st.subheader('Daily Selling')

col1, col2 = st.columns(2)

with col1:
    total_orders = penjualan_harian_df.order_count.sum()
    st.metric("Total of Selling", value=total_orders)
    
with col2:
    total_revenue = format_currency(penjualan_harian_df.revenue.sum(), "AUD", locale='es_CO')
    st.metric("Total of Revenue", value=total_revenue)

# Plot the line chart with data labels
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    penjualan_harian_df["order_date"],
    penjualan_harian_df["order_count"],
    marker='o',
    linewidth=2,
    color="#5c4219"
)

# Add data labels to the line chart
for i, value in enumerate(penjualan_harian_df["order_count"]):
    ax.text(penjualan_harian_df["order_date"].iloc[i], value + 0.5, str(value), color='black', fontsize=12, ha='center')

ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

# Display the chart in Streamlit
st.pyplot(fig)

st.subheader('Customer Count per Age Group (Line Chart)')

# Create age group data for line chart (Adults, Seniors, Youth)
adults_df = main_df[main_df["age_group"] == "Adults"].resample(rule='D', on='order_date').agg({
    "customer_id": "nunique"
}).rename(columns={"customer_id": "Adults"}).reset_index()

seniors_df = main_df[main_df["age_group"] == "Seniors"].resample(rule='D', on='order_date').agg({
    "customer_id": "nunique"
}).rename(columns={"customer_id": "Seniors"}).reset_index()

youth_df = main_df[main_df["age_group"] == "Youth"].resample(rule='D', on='order_date').agg({
    "customer_id": "nunique"
}).rename(columns={"customer_id": "Youth"}).reset_index()

# Merge the age group dataframes on order_date
umur_df = pd.merge(adults_df, seniors_df, on="order_date", how="outer")
umur_df = pd.merge(umur_df, youth_df, on="order_date", how="outer")
umur_df.fillna(0, inplace=True)

# Plot the line chart for age group counts
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(umur_df["order_date"], umur_df["Adults"], label="Adults", marker='o', linewidth=2, color="#C68E17")
ax.plot(umur_df["order_date"], umur_df["Seniors"], label="Seniors", marker='o', linewidth=2, color="#7C0A02")
ax.plot(umur_df["order_date"], umur_df["Youth"], label="Youth", marker='o', linewidth=2, color="#6495ED")

# Add data labels for each age group
for i, value in enumerate(umur_df["Adults"]):
    ax.text(umur_df["order_date"].iloc[i], value + 0.5, str(int(value)), color='black', fontsize=12, ha='center')
    
for i, value in enumerate(umur_df["Seniors"]):
    ax.text(umur_df["order_date"].iloc[i], value + 0.5, str(int(value)), color='black', fontsize=12, ha='center')
    
for i, value in enumerate(umur_df["Youth"]):
    ax.text(umur_df["order_date"].iloc[i], value + 0.5, str(int(value)), color='black', fontsize=12, ha='center')

# Add average age descriptions 
ax.text(0.02, 0.95, 'Adults: Age Between = 25-64', transform=ax.transAxes, fontsize=15, color="#ff6347", ha='left', va='top')
ax.text(0.02, 0.90, 'Seniors: Age Between = 65-80', transform=ax.transAxes, fontsize=15, color="#4682b4", ha='left', va='top')
ax.text(0.02, 0.85, 'Youth: Age Between = 20-24', transform=ax.transAxes, fontsize=15, color="#5c4219", ha='left', va='top')

ax.set_title("Customer Counts by Age Group over Time", fontsize=20)
ax.set_xlabel("Order Date", fontsize=15)
ax.set_ylabel("Customer Count", fontsize=15)

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=12)

st.pyplot(fig)



st.subheader("Items with the most and least sales")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))

colors = ["#4287f5", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

# Plot the top 5 products
sns.barplot(x="quantity_x", y="product_name", data=sum_penjualan_barang_df.head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Jumlah Penjualan", fontsize=30)
ax[0].set_title("Produk Paling Laris", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)

# Add data labels for the first plot
for i, v in enumerate(sum_penjualan_barang_df.head(5)["quantity_x"]):
    ax[0].text(v + 1, i, str(v), color='black', fontsize=30, va='center')

# Plot the worst 5 products
sns.barplot(x="quantity_x", y="product_name", data=sum_penjualan_barang_df.tail(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Jumlah Penjualan", fontsize=30)
ax[1].set_title("Produk Kurang Laris", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

# Add data labels for the second plot
for i, v in enumerate(sum_penjualan_barang_df.tail(5)["quantity_x"]):
    ax[1].text(v + 1, i, str(v), color='black', fontsize=30, va='center')

st.pyplot(fig)

st.subheader("Customer Distribution")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    
    sns.barplot(
        y="customer_count",
        x="gender",
        data=byjk_df.sort_values(by="customer_count", ascending=False),
        palette=colors,
        ax=ax
    )
    
    # Adding data labels
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2,  # X-coordinate (center of bar)
            p.get_height(),  # Y-coordinate (top of the bar)
            int(p.get_height()),  # Label text
            ha="center", va="bottom", fontsize=25  # Text alignment and size
        )
    
    ax.set_title("Number of Customers by Gender", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    st.pyplot(fig)

    
with col1:
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#D3D3D3", "#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    
    sns.barplot(
        y="customer_count",
        x="age_group",
        data=byumur_df.sort_values(by="customer_count", ascending=False),
        palette=colors,
        ax=ax
    )
    
    # Adding data labels
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height(),
            int(p.get_height()),
            ha="center", va="bottom", fontsize=25
        )
    
    ax.set_title("Number of Customers by Age", loc="center", fontsize=50)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=30)
    st.pyplot(fig)

    
with col2:    
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    
    sns.barplot(
        x="customer_count", 
        y="state",
        data=bydaerah_df.sort_values(by="customer_count", ascending=False),
        palette=colors,
        ax=ax
    )
    
    # Adding data labels
    for p in ax.patches:
        ax.text(
            p.get_width(),  # X-coordinate (width of the bar)
            p.get_y() + p.get_height() / 2,  # Y-coordinate (center of the bar)
            int(p.get_width()),  # Label text
            ha="left", va="center", fontsize=15  # Text alignment and size
        )
    
    ax.set_title("Number of Customer by States", loc="center", fontsize=30)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=15)
    st.pyplot(fig)
    
    
st.subheader('Most Sold Sizes (Column Chart)')

# Filter data untuk mengecualikan ukuran dengan huruf kecil ('l', 'm', 'xl')
main_df_filtered = main_df[~main_df['size'].isin(['l', 'm', 'xl'])]

# Agregasi data penjualan berdasarkan ukuran
size_sales_df = main_df_filtered.groupby('size').agg({
    "customer_id": "nunique"  # Menghitung jumlah customer unik
}).rename(columns={"customer_id": "Sales Count"}).reset_index()

# Urutkan data berdasarkan sales count secara descending
size_sales_df.sort_values(by="Sales Count", ascending=False, inplace=True)

# Plot chart kolom
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="size", y="Sales Count", data=size_sales_df, palette="coolwarm", ax=ax)

# Tambahkan label data ke tiap bar
for i in ax.containers:
    ax.bar_label(i, fmt='%d', label_type='edge', fontsize=12)

# Pengaturan judul dan label
ax.set_title("Most Sold Sizes", fontsize=20)
ax.set_xlabel("Size", fontsize=15)
ax.set_ylabel("Sales Count", fontsize=15)

# Pengaturan ukuran label pada sumbu x dan y
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Tampilkan grafik
st.pyplot(fig)



st.subheader('Most and Least Sold Product Types (Comparison)')

# Aggregate product type sales data
product_type_sales_df = main_df.groupby('product_type').agg({
    "customer_id": "nunique"
}).rename(columns={"customer_id": "Sales Count"}).reset_index()

# Find the most sold and least sold product types
most_sold = product_type_sales_df.loc[product_type_sales_df["Sales Count"].idxmax()]
least_sold = product_type_sales_df.loc[product_type_sales_df["Sales Count"].idxmin()]

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    "Product Type": [most_sold["product_type"], least_sold["product_type"]],
    "Sales Count": [most_sold["Sales Count"], least_sold["Sales Count"]],
    "Type": ["Most Sold", "Worst Selling"]  # Changed to 'Worst Selling'
})

# Plot the comparison
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Sales Count", y="Product Type", hue="Type", data=comparison_df, palette=["#5c4219", "#ff6347"], ax=ax)

# Add data labels to each bar
for i in ax.containers:
    ax.bar_label(i, fmt='%d', label_type='edge', fontsize=12)

ax.set_title("Comparison of Most and Worst Selling Product Types", fontsize=20)
ax.set_xlabel("Sales Count", fontsize=15)
ax.set_ylabel("Product Type", fontsize=15)

ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(title="Sales Type", fontsize=12)

st.pyplot(fig)


# Sales Count by Colour
st.subheader('Sales Count by Colour')

# Cek apakah ada missing values atau data duplikat
main_df = main_df.dropna(subset=['customer_id', 'colour'])  # Hapus baris dengan missing values
main_df = main_df.drop_duplicates()  # Hapus baris duplikat

# Agregasi data penjualan berdasarkan warna, menggunakan 'nunique' untuk menghitung customer_id unik
colour_sales_df = main_df.groupby('colour').agg({
    "customer_id": "nunique"  # Menghitung jumlah customer unik
}).rename(columns={"customer_id": "Sales Count"}).reset_index()

# Urutkan DataFrame berdasarkan 'Sales Count' secara descending
colour_sales_df = colour_sales_df.sort_values(by="Sales Count", ascending=False)

# Buat plot bar untuk sales count berdasarkan warna
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Sales Count", y="colour", data=colour_sales_df, palette="viridis", ax=ax)

# Menambahkan label pada setiap bar
for index, value in enumerate(colour_sales_df['Sales Count']):
    ax.text(value, index, str(value), color='black', va='center')

# Pengaturan judul dan label
ax.set_title("Sales Count by Colour", fontsize=20)
ax.set_xlabel("Sales Count", fontsize=15)
ax.set_ylabel("Colour", fontsize=15)

# Pengaturan ukuran label pada sumbu x dan y
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)

# Tampilkan grafik
st.pyplot(fig)


# Best Customer Based on RFM Parameters
st.subheader("Best Customer Based on RFM Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (days)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)

with col3:
    avg_frequency = format_currency(rfm_df.monetary.mean(), "AUD", locale='es_CO') 
    st.metric("Average Monetary", value=avg_frequency)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9"]

sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("customer_id", fontsize=30)
ax[0].set_title("By Recency (days)", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=30)
ax[0].tick_params(axis='x', labelsize=35)

sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("customer_id", fontsize=30)
ax[1].set_title("By Frequency", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=30)
ax[1].tick_params(axis='x', labelsize=35)

sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), palette=colors, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("customer_id", fontsize=30)
ax[2].set_title("By Monetary", loc="center", fontsize=50)
ax[2].tick_params(axis='y', labelsize=30)
ax[2].tick_params(axis='x', labelsize=35)

st.pyplot(fig)



# Create city mapping
city_mapping_df = create_city_mapping(main_df)

# Get the top 5 cities by customer count
top_cities_df = city_mapping_df.nlargest(5, 'customer_count')

# Display Top 5 City Mapping Chart
st.subheader('Top 5 Cities with the Largest Number of Customers')

# Create a bar chart for customer count by city
fig_city, ax_city = plt.subplots(figsize=(12, 6))
sns.barplot(x='customer_count', y='city', data=top_cities_df, ax=ax_city, palette='viridis')

ax_city.set_title('Top 5 Cities with the Largest Number of Customers', fontsize=18)
ax_city.set_xlabel('Number of Customers', fontsize=14)
ax_city.set_ylabel('City', fontsize=14)

# Add data labels above each bar
for index, value in enumerate(top_cities_df['customer_count']):
    ax_city.text(value, index, str(value), color='black', va='center')

# Display the top cities mapping chart in Streamlit
st.pyplot(fig_city)

# Sentiment and Wordcloud Analysis
st.subheader('Word Cloud and Sentiment Analysis')


# Footer
st.write('Built by Salman Fadhilurrohman © 2024 Dicoding Submission')

