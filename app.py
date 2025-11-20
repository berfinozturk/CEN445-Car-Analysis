import streamlit as st  # Streamlit kütüphanesini yükler
import pandas as pd  # Veri analizi için Pandas
import plotly.express as px  # Grafikler için Plotly
from sklearn.cluster import KMeans  # Makine öğrenmesi için K-Means

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Used Car Analysis", layout="wide")

# --- 1. VERİ ÖN İŞLEME (DATA PREPROCESSING) ---
@st.cache_data
def load_data():
    df = pd.read_csv("vehicles.csv")
    
    # Sütun isimlerini küçük harfe çevir
    df.columns = df.columns.str.lower()
    
    # Eksik verileri temizle
    df = df.dropna()
    
    # Veri tiplerini sayısal hale getir
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
    
    # Aykırı Değerleri Temizle (Outlier Removal)
    df = df[(df['price'] > 500) & (df['price'] < 500000)]
    df = df[df['year'] > 1990]
    df = df[df['odometer'] < 500000]
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- SIDEBAR (FİLTRELER) ---
st.sidebar.header("Dashboard Filters")

# Yıl Filtresi
min_year, max_year = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (2010, 2020))

# Marka Filtresi
all_brands = sorted(df['manufacturer'].unique())
selected_brands = st.sidebar.multiselect("Select Brands", all_brands, default=all_brands[:5])

# Filtreleme İşlemi
filtered_df = df[(df['year'].between(*year_range)) & (df['manufacturer'].isin(selected_brands))]

# --- BAŞLIK VE GİRİŞ ---
st.title("🚗 Used Car Price Analysis Dashboard")
st.markdown("""
This project is designed to analyze price dynamics in the used car market. 
The dataset has been cleaned and presented with interactive visualizations.
""")
# "Gösterilen Veri Sayısı" -> İngilizce
st.info(f"Number of Records Displayed: {len(filtered_df)} (Filtered)")

# --- SEKMELER (TABS) ---
tab1, tab2, tab3 = st.tabs(["Ali Sait Öz (Hierarchy)", "Berfin Öztürk (Trends)", "Arda Murat Abay (ML & Stats)"])

# =============================================================================
# TAB 1: ALİ SAİT ÖZ
# =============================================================================
with tab1:
    st.header("Categorical and Hierarchical Analysis - Ali Sait Öz")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Market Share by Brand (Treemap)")
        # Title: "Marka ve Model Pazar Payı" -> İngilizce
        fig_treemap = px.treemap(filtered_df, path=['manufacturer', 'model'], values='price', color='price',
                                 color_continuous_scale='RdBu', title="Market Share by Brand and Model")
        st.plotly_chart(fig_treemap, use_container_width=True)
        
    with col2:
        st.subheader("2. Hierarchy: Brand > Fuel > Transmission")
        # Title: "Marka - Yakıt - Vites Dağılımı" -> İngilizce
        fig_sunburst = px.sunburst(filtered_df.head(5000), path=['manufacturer', 'fuel', 'transmission'], 
                                   title="Distribution of Brand - Fuel - Transmission")
        st.plotly_chart(fig_sunburst, use_container_width=True)

    st.subheader("3. Top 10 Most Expensive Models (Bar Chart)")
    top_expensive = filtered_df.groupby('model')['price'].mean().sort_values(ascending=False).head(10).reset_index()
    # Title: "Ortalama Fiyatı En Yüksek 10 Model" -> İngilizce
    fig_bar = px.bar(top_expensive, x='price', y='model', orientation='h', title="Top 10 Models with Highest Average Price")
    st.plotly_chart(fig_bar, use_container_width=True)

# =============================================================================
# TAB 2: BERFİN ÖZTÜRK
# =============================================================================
with tab2:
    st.header("Trend and Time Series Analysis - Berfin Öztürk")
    
    st.subheader("4. Price vs Mileage Evolution over Time (Animation)")
    # Caption: "Play butonuna basarak..." -> İngilizce
    st.caption("Press the Play button to watch the evolution over the years.")
    
    anim_df = filtered_df.sort_values('year')
    # Title: "Yıllara Göre KM ve Fiyat Değişimi" -> İngilizce
    fig_anim = px.scatter(anim_df, x="odometer", y="price", animation_frame="year", 
                          color="manufacturer", size_max=60, range_x=[0,300000], range_y=[0,100000],
                          title="Evolution of Price vs Mileage Over Years")
    st.plotly_chart(fig_anim, use_container_width=True)
    
    st.subheader("5. Parallel Coordinates Plot")
    # Caption: "Fiyat, Yıl ve KM arasındaki..." -> İngilizce
    st.caption("Multidimensional relationship between Price, Year, and Odometer.")
    
    # Title: "Çoklu Değişken Analizi" -> İngilizce
    fig_parallel = px.parallel_coordinates(filtered_df.head(500), dimensions=['price', 'year', 'odometer'],
                                           color="price", title="Multivariate Analysis (First 500 Cars)")
    st.plotly_chart(fig_parallel, use_container_width=True)

    st.subheader("6. Average Price Trend (Line Chart)")
    yearly_trend = filtered_df.groupby('year')['price'].mean().reset_index()
    # Title: "Yıllara Göre Ortalama Fiyat..." -> İngilizce
    fig_line = px.line(yearly_trend, x='year', y='price', title="Average Price Change Over Years")
    st.plotly_chart(fig_line, use_container_width=True)

# =============================================================================
# TAB 3: ARDA MURAT ABAY
# =============================================================================
with tab3:
    st.header("Statistical Analysis and ML - Arda Murat Abay")
    
    st.subheader("7. K-Means Clustering (ML Segmentation)")
    # Açıklama metni -> İngilizce
    st.write("We segment cars into 3 categories (Economy, Mid-range, Luxury) based on Price and Odometer features.")
    
    ml_df = filtered_df[['price', 'odometer']].dropna()
    
    if len(ml_df) > 0:
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        ml_df['cluster'] = kmeans.fit_predict(ml_df)
        ml_df['cluster'] = ml_df['cluster'].astype(str)
        
        # Title ve Label -> İngilizce
        fig_cluster = px.scatter(ml_df, x='odometer', y='price', color='cluster', 
                                 title="Car Segmentation (Clustering Analysis)",
                                 labels={'cluster': 'Segment'})
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.warning("Not enough data for clustering.")

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("8. Price vs Odometer Density (Heatmap)")
        # Title: "Fiyat ve KM Yoğunluk Haritası" -> İngilizce
        fig_heatmap = px.density_heatmap(filtered_df, x="odometer", y="price", nbinsx=20, nbinsy=20, 
                                         title="Price and Odometer Density Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    with col4:
        st.subheader("9. Price Distribution by Fuel Type")
        # Title: "Yakıt Tipine Göre..." -> İngilizce
        fig_box = px.box(filtered_df, x="fuel", y="price", color="fuel", title="Price Distribution by Fuel Type")
        st.plotly_chart(fig_box, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("CEN445 Project - 2025 | Github Repository: [https://github.com/berfinozturk/CEN445-Car-Analysis]")