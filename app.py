# Gerekli kütüphaneleri içe aktarıyoruz.
import streamlit as st # Web uygulamasını oluşturmak için ana kütüphane.
import pandas as pd # Veri manipülasyonu ve analizi için (DataFrames).
import plotly.express as px # Etkileşimli ve güzel görünümlü grafikler oluşturmak için.
from sklearn.cluster import KMeans # Makine öğrenimi bölümü için kümeleme algoritması.

# -----------------------------------------------------------------------------

# SAYFA AYARLARI
# st.set_page_config() fonksiyonu, tarayıcı sekmesinin başlığını ve sayfa düzenini ayarlar.
st.set_page_config(page_title="Used Car Analysis", layout="wide") 

# SUNUM YORUMU: "Projemiz, kullanıcı deneyimini optimize etmek için sayfayı geniş (wide) düzende ayarlayarak görsellerin daha ferah görünmesini sağladık."

# -----------------------------------------------------------------------------

# 1. VERİ YÜKLEME VE ÖN İŞLEME
# @st.cache_data dekoratörü, veriyi sadece bir kez yüklemeyi ve ön işlemeyi garanti eder.
# Bu, uygulamanın performansını artırır ve kullanıcı filtre değiştirse bile verinin tekrar tekrar okunmasını engeller.
@st.cache_data
def load_data():
    # Veri setini yüklüyoruz.
    df = pd.read_csv("vehicles.csv")
    
    # Sütun isimlerini düzenle: Hepsi küçük harfe çevrilir ve baştaki/sondaki boşluklar temizlenir.
    df.columns = df.columns.str.lower().str.strip()
    
    # SUNUM YORUMU: "Farklı veri setlerinden kaynaklanabilecek tutarsızlıkları gidermek için, veri kalitesini artırma adına sütun adlarını standartlaştırdık."
    
    # Sütun eşleştirme (Renaming): Farklı veri setlerindeki muhtemel farklı isimleri standart hale getiriyoruz.
    renames = {
        'make': 'manufacturer',
        'brand': 'manufacturer',
        'company': 'manufacturer',
        'mileage': 'odometer',
        'kms_driven': 'odometer',
        'fueltype': 'fuel',
        'fuel_type': 'fuel',
        'transmission_type': 'transmission',
        'model_year': 'year'
    }
    df = df.rename(columns=renames)
    
    # Eğer "manufacturer" sütunu hala yoksa, ilk metin sütununu 'manufacturer' olarak adlandırır. (Bu bir yedek çözümdür.)
    if 'manufacturer' not in df.columns:
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            df = df.rename(columns={text_cols[0]: 'manufacturer'})

    # Eksik verileri temizle: Şimdilik tüm satırlarda eksik değer içerenleri çıkarıyoruz.
    df = df.dropna()
    
    # SUNUM YORUMU: "Analizimizin doğruluğu için eksik değer içeren tüm satırları temizledik."
    
    # Veri tiplerini düzelt: Özellikle fiyat (price) sütununda temizlik yapılması gerekiyor.
    if 'price' in df.columns and df['price'].dtype == 'object':
        # '$' ve ',' gibi sayısal olmayan karakterleri temizliyoruz.
        df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True)
        
    # İlgili sütunları sayısal (numeric) veri tipine dönüştürüyoruz. Hata veren değerler NaN olur (coerce).
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
    
    # Sayısala çevirme sonrası oluşan NaN (boş) değerleri temizliyoruz.
    df = df.dropna(subset=['price', 'year', 'odometer'])

    # AYKIRI DEĞER TEMİZLİĞİ (OUTLIER REMOVAL)
    # SUNUM YORUMU: "Veri setindeki olası hataları ve aykırı değerleri temizleyerek analizlerimizin gerçekçi olmasını sağladık."
    
    # Fiyat temizliği: Çok ucuz veya çok pahalı olan araçları çıkarıyoruz (realistik aralık).
    df = df[(df['price'] > 500) & (df['price'] < 500000)]
    
    # YIL FİLTRESİ: Analizi anlamlı bir aralığa (1990-2020) indiriyoruz.
    df = df[(df['year'] >= 1990) & (df['year'] <= 2020)]
    
    # KM temizliği: Çok yüksek kilometreye sahip araçları çıkarıyoruz (örn. 500.000 km üstü).
    df = df[df['odometer'] < 500000]
    
    return df

# Veri yükleme işlemini try-except bloğu ile güvenli hale getiriyoruz.
try:
    df = load_data()
except Exception as e:
    # Hata durumunda Streamlit'e hata mesajı gösterip uygulamayı durdurur.
    st.error(f"Error loading data: {e}")
    st.stop()

# -----------------------------------------------------------------------------

# SIDEBAR (FİLTRELER)
st.sidebar.header("Dashboard Filters")

# Yıl Aralığı Slider'ı: Kullanıcının filtreleme yapmasını sağlar.
# Varsayılan değer olarak 2010 ve 2020 arası seçili gelir.
year_range = st.sidebar.slider("Select Year Range", 1990, 2020, (2010, 2020))

# Marka seçimi için tüm markaların ve en popüler 5 markanın listesini hazırlıyoruz.
all_brands = sorted(df['manufacturer'].unique())
popular_brands = df['manufacturer'].value_counts().head(5).index.tolist()

# Marka seçimi Multiselect widget'ı
col1, col2, col3 = st.sidebar.columns([2, 1, 1])
with col1:
    # Varsayılan olarak en popüler 5 marka seçili gelir.
    selected_brands = st.sidebar.multiselect("Select Brands", all_brands, default=popular_brands)
with col2:
    # "Popular 5" butonu ile ilk 5 markayı seçme kolaylığı sunulur.
    if st.sidebar.button("Popular 5", help="Select top 5 popular brands"):
        st.session_state.selected_brands = popular_brands # Seçimi session state'e kaydedip
        st.rerun() # Sayfayı yeniden yükleriz (rerun).
with col3:
    # "All" butonu ile tüm markaları seçme kolaylığı sunulur.
    if st.sidebar.button("All", help="Select all brands"):
        st.session_state.selected_brands = all_brands
        st.rerun()# Update selected_brands from session state if button was clicked

# Butona tıklandığında session state'ten seçimi alıp state'i temizler.
if 'selected_brands' in st.session_state:
    selected_brands = st.session_state.selected_brands
    del st.session_state.selected_brands

# Filtreleme: Seçilen yıl aralığı ve markalara göre ana veri setini filtreleriz.
if selected_brands:
    filtered_df = df[(df['year'].between(*year_range)) & (df['manufacturer'].isin(selected_brands))]
else:
    # Marka seçilmezse sadece yıl aralığına göre filtreleme yapılır.
    filtered_df = df[df['year'].between(*year_range)]

# -----------------------------------------------------------------------------

# BAŞLIK VE GİRİŞ
st.title("🚗 Used Car Price Analysis Dashboard")
st.markdown("""
This project is designed to analyze price dynamics in the used car market. 
The dataset has been cleaned and presented with interactive visualizations.
""")
# Filtreleme sonrası kaç kaydın gösterildiğini bilgi kutusunda gösterir.
st.info(f"Number of Records Displayed: {len(filtered_df)} (Filtered)")

# SUNUM YORUMU: "Filtreleme mekanizması sayesinde, gösterge tablosunun dinamik olarak çalıştığını ve anlık kayıt sayısını görebilirsiniz."

# -----------------------------------------------------------------------------

# SEKMELER (TABS)
# Analizleri farklı kategorilerde gruplandırmak için sekmeler oluşturuyoruz.
tab1, tab2, tab3 = st.tabs(["Hierarchical Analysis", "Trend Analysis", "ML & Stats"])

# -----------------------------------------------------------------------------

# TAB 1: KATEGORİK VE HİYERARŞİK ANALİZ
with tab1:
    st.header("Categorical and Hierarchical Analysis")
    
    col1, col2 = st.columns(2) # İlk iki grafiği yan yana yerleştirmek için sütunlar oluşturuyoruz.
    
    with col1:
        st.subheader("1. Market Share by Brand & Transmission")
        # Treemap (Ağaç Haritası): Marka ve vites tipine göre pazar payını (fiyata göre) gösterir.
        fig_treemap = px.treemap(filtered_df, path=['manufacturer', 'transmission'], values='price', color='price',
                                 color_continuous_scale='RdBu', title="Market Share by Brand and Transmission")
        st.plotly_chart(fig_treemap, use_container_width=True)
        
        # SUNUM YORUMU: "Treemap, hangi markanın/modelin toplam fiyat hacminde ne kadar yer kapladığını ve bu payın manuel/otomatik vites arasında nasıl bölündüğünü görselleştirir. Renk, aracın ortalama fiyatını gösterir."

    with col2:
        st.subheader("2. Hierarchy: Brand > Fuel > Transmission")
        # Sunburst (Güneş Işını Grafiği): Marka, yakıt ve vitesin iç içe hiyerarşisini gösterir.
        # Büyük veri setleri için performansı korumak adına ilk 5000 kayıtla sınırlanmıştır.
        fig_sunburst = px.sunburst(filtered_df.head(5000), path=['manufacturer', 'fuel', 'transmission'], 
                                     title="Distribution of Brand - Fuel - Transmission")
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # SUNUM YORUMU: "Sunburst grafiği ile, belirli bir markanın önce hangi yakıt tipine, ardından hangi vites tipine ayrıldığını hiyerarşik olarak inceliyoruz. Bu, pazar segmentasyonunu anlamak için kritik öneme sahiptir."

    st.subheader("3. Average Price by Brand")
    # Markalara göre ortalama fiyatları hesaplar ve en yüksek 10'u alır.
    top_expensive = filtered_df.groupby('manufacturer')['price'].mean().sort_values(ascending=False).head(10).reset_index()
    # Yatay çubuk grafik ile en pahalı 10 markayı görselleştiririz.
    fig_bar = px.bar(top_expensive, x='price', y='manufacturer', orientation='h', title="Top 10 Brands with Highest Average Price")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # SUNUM YORUMU: "Bu çubuk grafik, portföyümüzdeki en yüksek ortalama fiyata sahip 10 markayı/modeli gösterir. Bu bilgi, kârlılık stratejilerimizi yönlendirmek için temel veridir."

# -----------------------------------------------------------------------------

# TAB 2 ve TAB 3 (Kodda İçerikleri Boş, ama Başlıkları Mevcut)
# Bu sekmeler şu anda sadece isimlendirilmiştir. (Sizin kodunuzda sadece başlıkları var.)
with tab2:
    st.header("Trend Analysis (Missing Content)")
    st.markdown("Placeholder for Time Series and Odometer/Price trends.")
    
with tab3:
    st.header("ML & Stats (Missing Content)")
    st.markdown("Placeholder for K-Means Clustering and other statistical summaries.")

# -----------------------------------------------------------------------------

# TAB 2: TREND ANALİZİ

with tab2:
    st.header("Trend and Time Series Analysis")
    
    st.subheader("4. Price vs Mileage Evolution over Time")
    st.caption("Press the Play button to watch the evolution over the years.")
    
    anim_df = filtered_df.sort_values('year')
    fig_anim = px.scatter(anim_df, x="odometer", y="price", animation_frame="year", 
                          color="manufacturer", size_max=60, range_x=[0,300000], range_y=[0,100000],
                          title="Evolution of Price vs Mileage Over Years")
    st.plotly_chart(fig_anim, use_container_width=True)
    
    st.subheader("5. Parallel Coordinates Plot")
    st.caption("Multidimensional relationship between Price, Year, and Odometer.")
    fig_parallel = px.parallel_coordinates(filtered_df.head(500), dimensions=['price', 'year', 'odometer'],
                                           color="price", title="Multivariate Analysis (First 500 Cars)")
    st.plotly_chart(fig_parallel, use_container_width=True)

    st.subheader("6. Average Price Trend")
    yearly_trend = filtered_df.groupby('year')['price'].mean().reset_index()
    fig_line = px.line(yearly_trend, x='year', y='price', title="Average Price Change Over Years")
    st.plotly_chart(fig_line, use_container_width=True)


# TAB 3: ML & İSTATİSTİK

with tab3:
    st.header("Statistical Analysis and ML")
    
    st.subheader("7. K-Means Clustering")
    st.write("We segment cars into 3 categories (Economy, Mid-range, Luxury) based on Price and Odometer features.")
    
    ml_df = filtered_df[['price', 'odometer']].dropna()
    
    if len(ml_df) > 0:
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
        ml_df['cluster'] = kmeans.fit_predict(ml_df)
        ml_df['cluster'] = ml_df['cluster'].astype(str)
        # Kümeleri isimlendirme 
        cluster_means = ml_df.groupby('cluster')['price'].mean().sort_values()
        cluster_map = {
            cluster_means.index[0]: 'Economy',
            cluster_means.index[1]: 'Mid-range',
            cluster_means.index[2]: 'Luxury'
        }
        ml_df['cluster'] = ml_df['cluster'].map(cluster_map)
        
        fig_cluster = px.scatter(ml_df, x='odometer', y='price', color='cluster', 
                                 title="Car Segmentation (Clustering Analysis)",
                                 labels={'cluster': 'Segment'}, color_discrete_map={"Economy": "blue", "Mid-range": "green", "Luxury": "red"})
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.warning("Not enough data for clustering.")

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("8. Price vs Odometer Density")
        fig_heatmap = px.density_heatmap(filtered_df, x="odometer", y="price", nbinsx=20, nbinsy=20, 
                                         title="Price and Odometer Density Heatmap")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    with col4:
        st.subheader("9. Price Distribution by Fuel Type")
        fig_box = px.box(filtered_df, x="fuel", y="price", color="fuel", title="Price Distribution by Fuel Type")
        st.plotly_chart(fig_box, use_container_width=True)

#FOOTER
st.markdown("---")
st.markdown("CEN445 Project - 2025 | Github Repository: [https://github.com/berfinozturk/CEN445-Car-Analysis]")




