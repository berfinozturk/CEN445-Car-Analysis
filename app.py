import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import itertools  # BU SATIR EKSİKTİ, ŞİMDİ EKLENDİ.

# -----------------------------------------------------------------------------
# SAYFA AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Used Car Analysis", layout="wide")

# -----------------------------------------------------------------------------
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("vehicles.csv")
    except FileNotFoundError:
        st.error("Veri dosyası (vehicles.csv) bulunamadı.")
        st.stop()
    
    df.columns = df.columns.str.lower().str.strip()
    
    renames = {
        'make': 'manufacturer', 'brand': 'manufacturer', 'company': 'manufacturer',
        'mileage': 'odometer', 'kms_driven': 'odometer',
        'fueltype': 'fuel', 'fuel_type': 'fuel',
        'transmission_type': 'transmission', 'model_year': 'year'
    }
    df = df.rename(columns=renames)
    
    if 'manufacturer' not in df.columns:
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            df = df.rename(columns={text_cols[0]: 'manufacturer'})

    df = df.dropna()
    
    if 'price' in df.columns and df['price'].dtype == 'object':
        df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True)
        
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
    
    df = df.dropna(subset=['price', 'year', 'odometer'])
    df = df[(df['price'] > 500) & (df['price'] < 500000)]
    df = df[(df['year'] >= 1990) & (df['year'] <= 2020)]
    df = df[df['odometer'] < 500000]
    
    # HER SATIR İÇİN BENZERSİZ KİMLİK
    df['unique_id'] = "car_" + df.index.astype(str)
    
    return df

df = load_data()

# -----------------------------------------------------------------------------
# SIDEBAR (FİLTRELER)
# -----------------------------------------------------------------------------
st.sidebar.header("Dashboard Filters")
year_range = st.sidebar.slider("Select Year Range", 1990, 2020, (2010, 2020))

all_brands = sorted(df['manufacturer'].unique())
popular_brands = df['manufacturer'].value_counts().head(5).index.tolist()

col1, col2, col3 = st.sidebar.columns([2, 1, 1])

if 'selected_brands_state' not in st.session_state:
    st.session_state.selected_brands_state = popular_brands

with col2:
    if st.sidebar.button("Pop 5"):
        st.session_state.selected_brands_state = popular_brands
        st.rerun()
        
with col3:
    if st.sidebar.button("All"):
        st.session_state.selected_brands_state = all_brands
        st.rerun()

with col1:
    selected_brands = st.sidebar.multiselect(
        "Select Brands", 
        all_brands, 
        default=st.session_state.selected_brands_state
    )

if selected_brands:
    filtered_df = df[(df['year'].between(*year_range)) & (df['manufacturer'].isin(selected_brands))]
else:
    filtered_df = df[df['year'].between(*year_range)]

# -----------------------------------------------------------------------------
# BAŞLIK
# -----------------------------------------------------------------------------
st.title("🚗 Used Car Price Analysis Dashboard")
st.info(f"Number of Records Displayed: {len(filtered_df)} (Filtered)")

# -----------------------------------------------------------------------------
# SEKMELER
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Hierarchical Analysis", "Trend Analysis", "ML & Stats"])

# TAB 1
with tab1:
    st.header("Categorical and Hierarchical Analysis")
    col_t1_1, col_t1_2 = st.columns(2)
    with col_t1_1:
        st.subheader("1. Market Share by Brand & Transmission")
        fig_treemap = px.treemap(filtered_df, path=['manufacturer', 'transmission'], values='price', color='price',
                                 color_continuous_scale='RdBu')
        st.plotly_chart(fig_treemap, theme="streamlit")
        
    with col_t1_2:
        st.subheader("2. Hierarchy: Brand > Fuel > Transmission")
        fig_sunburst = px.sunburst(filtered_df.head(5000), path=['manufacturer', 'fuel', 'transmission'])
        st.plotly_chart(fig_sunburst, theme="streamlit")

    st.subheader("3. Average Price by Brand")
    top_expensive = filtered_df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(10).reset_index()
    fig_bar = px.bar(top_expensive, x='price', y='manufacturer', orientation='h')
    st.plotly_chart(fig_bar, theme="streamlit")

# TAB 2
with tab2:
    st.header("Trend and Time Series Analysis")
    st.subheader("4. Price vs Mileage Evolution over Time")
    st.caption("Now both 'All Dots' and 'All Legends' work simultaneously.")
    
    anim_df = filtered_df.sort_values('year')
    
    if not anim_df.empty:
        years = list(range(int(anim_df['year'].min()), int(anim_df['year'].max()) + 1))
        target_brands = selected_brands if selected_brands else anim_df['manufacturer'].unique()
        
        # itertools hatası vermemesi için yukarıda import edildi
        skeleton = pd.DataFrame(list(itertools.product(years, target_brands)), columns=['year', 'manufacturer'])
        skeleton['unique_id'] = "dummy_" + skeleton['year'].astype(str) + "_" + skeleton['manufacturer']
        
        final_anim_df = pd.concat([anim_df, skeleton], ignore_index=True)
        final_anim_df = final_anim_df.sort_values(['year', 'manufacturer'])
        
        brands_order = sorted(target_brands)

        fig_anim = px.scatter(
            final_anim_df, 
            x="odometer", 
            y="price", 
            animation_frame="year", 
            animation_group="unique_id", 
            color="manufacturer", 
            opacity=0.6,
            size_max=40, 
            range_x=[0, 350000], 
            range_y=[0, 150000],
            category_orders={"manufacturer": brands_order},
            title="Evolution of Price vs Mileage Over Years",
            hover_data=['manufacturer', 'price', 'odometer']
        )
        
        st.plotly_chart(fig_anim, theme="streamlit")
        
    else:
        st.warning("No data available for animation.")

    st.subheader("5. Parallel Coordinates Plot")
    fig_parallel = px.parallel_coordinates(filtered_df.head(500), dimensions=['price', 'year', 'odometer'],
                                             color="price", title="Multivariate Analysis (First 500 Cars)")
    
    # Margin ayarı (Sol ve Üst boşluk)
    fig_parallel.update_layout(margin=dict(l=60, r=20, t=100, b=20))
    
    st.plotly_chart(fig_parallel, theme="streamlit")

    # TAB 2 İÇİNE ALINDI
    st.subheader("6. Average Price Trend")
    
    brand_trend = filtered_df.groupby(['year', 'manufacturer'])['price'].mean().reset_index()
    
    fig_line = px.line(
        brand_trend, 
        x='year', 
        y='price', 
        color='manufacturer', 
        title="Average Price Change Over Years (By Brand)"
    )
    
    st.plotly_chart(fig_line, theme="streamlit")

# TAB 3
with tab3:
    st.header("Statistical Analysis and ML")
    st.subheader("7. K-Means Clustering")
    
    ml_df = filtered_df[['price', 'odometer']].dropna()
    if len(ml_df) > 10:
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        ml_df['cluster'] = kmeans.fit_predict(ml_df)
        ml_df['cluster'] = ml_df['cluster'].astype(str)
        cluster_means = ml_df.groupby('cluster')['price'].mean().sort_values()
        map_dict = {cluster_means.index[0]: 'Economy', cluster_means.index[1]: 'Mid-range', cluster_means.index[2]: 'Luxury'}
        ml_df['segment'] = ml_df['cluster'].map(map_dict)
        
        fig_cluster = px.scatter(ml_df, x='odometer', y='price', color='segment', 
                                 color_discrete_map={"Economy": "blue", "Mid-range": "green", "Luxury": "red"})
        st.plotly_chart(fig_cluster, theme="streamlit")
    else:
        st.warning("Not enough data.")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("8. Price vs Odometer Density")
        fig_heatmap = px.density_heatmap(filtered_df, x="odometer", y="price", nbinsx=20, nbinsy=20)
        st.plotly_chart(fig_heatmap, theme="streamlit")
    with col4:
        st.subheader("9. Price Distribution by Fuel Type")
        fig_box = px.box(filtered_df, x="fuel", y="price", color="fuel")
        st.plotly_chart(fig_box, theme="streamlit")

st.markdown("---")
st.markdown("CEN445 Project - 2025")
