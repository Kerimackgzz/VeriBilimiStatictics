# %% [markdown]
# # ISE 216 - Veri Bilimi İçin İstatistik Ödevi
# Bu dosya `addiction_population_data.csv` veri seti için uçtan uca veri temizleme,
# tanımlayıcı istatistik, sınırlı görselleştirme, normallik testleri, olasılık
# dağılımları, Merkezi Limit Teoremi, güven aralıkları ve 3 hipotez testi içerir.

# %% 
# Gerekli kütüphaneleri içe aktar
from pathlib import Path
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

sns.set_theme(style="whitegrid", palette="Set2")

CSV_PATH = Path("addiction_population_data.csv")
FIG_DIR = Path("figures")
OUT_DIR = Path("outputs")
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# %% 
# Yardımcı fonksiyonları tanımla
def print_section(title: str) -> None:
    """Konsol çıktısını bölümlere ayırır."""
    line = "=" * 90
    print(f"\n{line}\n{title}\n{line}")


def save_current_figure(filename: str) -> None:
    """Aktif matplotlib grafiğini kaydeder ve kapatır."""
    path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Grafik kaydedildi: {path}")


def classify_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Sütunları veri tipi ve istatistiksel değişken türüne göre özetler."""
    rows = []
    for col in dataframe.columns:
        dtype = str(dataframe[col].dtype)
        if pd.api.types.is_numeric_dtype(dataframe[col]) and not pd.api.types.is_bool_dtype(dataframe[col]):
            variable_type = "Sayısal"
        else:
            variable_type = "Kategorik"
        rows.append(
            {
                "sütun": col,
                "veri_tipi": dtype,
                "istatistiksel_tür": variable_type,
                "benzersiz_değer_sayısı": dataframe[col].nunique(dropna=True),
            }
        )
    return pd.DataFrame(rows)


def get_categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    """Pandas sürüm uyarısı üretmeden kategorik/metinsel sütunları bulur."""
    categorical_columns = []
    for col in dataframe.columns:
        series = dataframe[col]
        if (
            pd.api.types.is_bool_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            categorical_columns.append(col)
    return categorical_columns


def skewness_comment(value: float) -> str:
    """Çarpıklık değerini Türkçe yorumlar."""
    if value > 0.5:
        return "Pozitif çarpık / sağa çarpık: sağ kuyruk daha uzundur."
    if value < -0.5:
        return "Negatif çarpık / sola çarpık: sol kuyruk daha uzundur."
    return "Yaklaşık simetrik dağılım."


def kurtosis_comment(value: float) -> str:
    """Fisher basıklık değerini Türkçe yorumlar."""
    if value > 0.5:
        return "Leptokurtik: normalden daha sivri/ağır kuyruklu."
    if value < -0.5:
        return "Platikurtik: normalden daha basık/hafif kuyruklu."
    return "Mezokurtik: normale yakın basıklık."


def normality_decision(p_value: float, alpha: float = 0.05) -> str:
    """Normallik p-değerine göre karar metni üretir."""
    if pd.isna(p_value):
        return "Test uygulanamadı"
    return "H0 reddedilir: normal dağılıma uygun değildir." if p_value < alpha else "H0 reddedilemez: normal dağılıma aykırı güçlü kanıt yoktur."


def hypothesis_decision(p_value: float, alpha: float = 0.05) -> str:
    """Hipotez testi p-değerine göre karar metni üretir."""
    return "H0 reddedilir" if p_value < alpha else "H0 reddedilemez"


def bool_series_to_int(series: pd.Series) -> pd.Series:
    """True/False değerlerini oran testleri için 1/0 formatına çevirir."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    return series.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})


def safe_norm_interval(confidence: float, loc: float, scale: float) -> tuple[float, float]:
    """SciPy sürüm farklarına uyumlu normal güven aralığı hesaplar."""
    try:
        return stats.norm.interval(confidence=confidence, loc=loc, scale=scale)
    except TypeError:
        return stats.norm.interval(confidence, loc=loc, scale=scale)


# %% [markdown]
# ## 1. Veriye Genel Bakış

# %%
# CSV dosyasını oku ve ilk genel çıktıları üret
print_section("1. VERİYE GENEL BAKIŞ")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"{CSV_PATH} bulunamadı. Script ile CSV aynı klasörde olmalıdır.")

df_raw = pd.read_csv(CSV_PATH)
df = df_raw.copy()

print("# df.head() çıktısı")
print(df.head())

print("\n# df.info() çıktısı")
info_buffer = io.StringIO()
df.info(buf=info_buffer)
print(info_buffer.getvalue())

print("# df.describe() çıktısı")
print(df.describe().T)

print("\n# Tüm sütunlar: veri tipi ve kategorik/sayısal bilgisi")
column_summary = classify_columns(df)
print(column_summary)
column_summary.to_csv(OUT_DIR / "01_column_summary.csv", index=False, encoding="utf-8-sig")

print(f"\nVeri seti boyutu: {df.shape[0]} satır x {df.shape[1]} sütun")
print("\nBeklenen yorum: Bu bölüm veri setinin satır-sütun yapısını, veri tiplerini vclee ilk gözlemleri gösterir.")
print("\n>>> TABLO: outputs/01_column_summary.csv  — sütun türleri ve benzersiz değer sayıları burada.")


# %% [markdown]
# ## 2. Veri Temizleme ve Ön İşleme

# %%
# Eksik değerleri tespit et, yüzdelerini hesapla ve uygun yöntemle doldur
print_section("2. VERİ TEMİZLEME VE ÖN İŞLEME")

missing_table = pd.DataFrame(
    {
        "eksik_sayı": df.isnull().sum(),
        "eksik_yüzde": df.isnull().mean() * 100,
    }
).sort_values("eksik_sayı", ascending=False)
print("# Eksik değer sayısı ve yüzdesi")
print(missing_table)
missing_table.to_csv(OUT_DIR / "02_missing_values.csv", encoding="utf-8-sig")

numeric_cols_all = [col for col in df.select_dtypes(include=[np.number]).columns if col != "id"]
categorical_cols_all = get_categorical_columns(df)

for col in numeric_cols_all:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols_all:
    mode_values = df[col].mode(dropna=True)
    if not mode_values.empty:
        df[col] = df[col].fillna(mode_values.iloc[0])

print("\nEksik değer doldurma sonrası toplam eksik değer:", int(df.isnull().sum().sum()))
print("Yorum: Sayısal değişkenlerde medyan, kategorik değişkenlerde mod kullanıldı.")
print("\n>>> TABLO: outputs/02_missing_values.csv  — hangi sütunda kaç eksik değer vardı.")


# %%
# Yinelenen satırları tespit et ve kaldır
duplicate_count = int(df.duplicated().sum())
print(f"\n# Yinelenen satır sayısı: {duplicate_count}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Yinelenen satırlar kaldırıldıktan sonra boyut: {df.shape[0]} satır x {df.shape[1]} sütun")


# %%
# Mantıksal hataları analiz et ve mümkün olanları düzelt
logical_errors = []

if {"age_started_smoking", "age"}.issubset(df.columns):
    invalid_smoking_age = df["age_started_smoking"] > df["age"]
    logical_errors.append(
        {
            "kontrol": "age_started_smoking > age",
            "hatalı_kayıt_sayısı": int(invalid_smoking_age.sum()),
            "yorum": "Sigara başlama yaşı kişinin mevcut yaşından büyük olamaz.",
        }
    )
    df.loc[invalid_smoking_age, "age_started_smoking"] = np.nan

if {"age_started_drinking", "age"}.issubset(df.columns):
    invalid_drinking_age = df["age_started_drinking"] > df["age"]
    logical_errors.append(
        {
            "kontrol": "age_started_drinking > age",
            "hatalı_kayıt_sayısı": int(invalid_drinking_age.sum()),
            "yorum": "Alkole başlama yaşı kişinin mevcut yaşından büyük olamaz.",
        }
    )
    df.loc[invalid_drinking_age, "age_started_drinking"] = np.nan

range_rules = {
    "age": (0, 120),
    "annual_income_usd": (0, None),
    "children_count": (0, None),
    "smokes_per_day": (0, None),
    "drinks_per_week": (0, None),
    "attempts_to_quit_smoking": (0, None),
    "attempts_to_quit_drinking": (0, None),
    "sleep_hours": (0, 24),
    "bmi": (10, 80),
}

for col, (lower, upper) in range_rules.items():
    if col not in df.columns:
        continue
    mask = pd.Series(False, index=df.index)
    if lower is not None:
        mask = mask | (df[col] < lower)
    if upper is not None:
        mask = mask | (df[col] > upper)
    logical_errors.append(
        {
            "kontrol": f"{col} mantıksal aralık kontrolü",
            "hatalı_kayıt_sayısı": int(mask.sum()),
            "yorum": f"{col} için beklenen aralık: {lower} - {upper if upper is not None else 'üst sınır yok'}.",
        }
    )
    df.loc[mask, col] = np.nan

if {"children_count", "age"}.issubset(df.columns):
    child_age_issue = (df["children_count"] > 0) & (df["age"] < 15)
    logical_errors.append(
        {
            "kontrol": "children_count > 0 ve age < 15",
            "hatalı_kayıt_sayısı": int(child_age_issue.sum()),
            "yorum": "15 yaş altı bireylerde çocuk sayısı pozitifse mantıksal tutarsızlık olabilir.",
        }
    )
    df.loc[child_age_issue, "children_count"] = np.nan

logical_error_table = pd.DataFrame(logical_errors)
print("\n# Mantıksal hata kontrolleri")
print(logical_error_table.to_string(index=False))
logical_error_table.to_csv(OUT_DIR / "02_logical_errors.csv", index=False, encoding="utf-8-sig")

for col in [c for c in numeric_cols_all if c in df.columns]:
    if col in ["age_started_smoking", "age_started_drinking"] and "age" in df.columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(np.minimum(median_value, df["age"]))
    else:
        df[col] = df[col].fillna(df[col].median())

print("\nMantıksal hata düzeltmeleri sonrası toplam eksik değer:", int(df.isnull().sum().sum()))
print("Yorum: Kişinin yaşından büyük başlama yaşları ve imkânsız aralıklar NaN yapılıp medyanla dolduruldu.")
print("\n>>> TABLO: outputs/02_logical_errors.csv  — hangi mantıksal kontrol kaç hata buldu.")


# %%
# IQR yöntemiyle aykırı değerleri tespit et
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != "id"]
outlier_rows = []

for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_rows.append(
        {
            "değişken": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "alt_sınır": lower_bound,
            "üst_sınır": upper_bound,
            "aykırı_sayı": int(outlier_mask.sum()),
            "aykırı_yüzde": outlier_mask.mean() * 100,
        }
    )

outlier_table = pd.DataFrame(outlier_rows).sort_values("aykırı_sayı", ascending=False)
print("\n# IQR aykırı değer özeti")
print(outlier_table)
outlier_table.to_csv(OUT_DIR / "02_outliers_iqr.csv", index=False, encoding="utf-8-sig")
print("\n>>> TABLO: outputs/02_outliers_iqr.csv  — her değişkende kaç aykırı değer var, Q1/Q3/IQR sınırları.")
print(
    "IQR neden kullanıldı? IQR, Q3-Q1 yani verinin ortadaki yüzde 50'lik kısmının genişliğidir. "
    "Aykırı değerleri ortalama ve standart sapmadan daha dayanıklı biçimde bulur; çünkü uç değerlerden daha az etkilenir. "
    "Bu ödevde IQR özeti, hangi değişkenlerde sıra dışı gözlemler olduğunu görmek ve sonraki istatistiksel yorumlarda dikkatli olmak için yapıldı."
)
print("Yorum: Aykırı değerler silinmedi; istatistiksel yorum için raporlandı.")


# %%
# Sınırlı sayıda temel değişken için aykırı değer boxplot görselleştirmesi yap
visual_numeric_cols = [col for col in ["age", "annual_income_usd", "smokes_per_day", "drinks_per_week", "sleep_hours", "bmi"] if col in df.columns]

if visual_numeric_cols:
    fig, axes = plt.subplots(1, len(visual_numeric_cols), figsize=(4.2 * len(visual_numeric_cols), 5))
    if len(visual_numeric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, visual_numeric_cols):
        sns.boxplot(y=df[col], ax=ax, color="#9ecae1")
        ax.set_title(f"{col} Box Plot")
        ax.set_xlabel("Değişken")
        ax.set_ylabel(col)
        ax.legend(handles=[plt.Line2D([0], [0], color="#9ecae1", lw=8, label=col)], loc="best")
    save_current_figure("02_iqr_boxplots.png")
    print("\n>>> GRAFİK: figures/02_iqr_boxplots.png  — 6 sayısal değişken için yan yana boxplot; aykırı değerleri görsel kontrol et.")


# %%
# Gerekli kategorik değişkenleri get_dummies ile encode et
identifier_cols = [col for col in ["id", "name"] if col in df.columns]
low_cardinality_cats = [
    col
    for col in get_categorical_columns(df)
    if col not in identifier_cols and df[col].nunique(dropna=True) <= 20
]

df_encoded = pd.get_dummies(df.drop(columns=identifier_cols, errors="ignore"), columns=low_cardinality_cats, drop_first=True)
print("\n# Encoding özeti")
print(f"Encode edilen düşük kardinaliteli kategorik sütunlar: {low_cardinality_cats}")
print(f"Encode edilmiş veri boyutu: {df_encoded.shape[0]} satır x {df_encoded.shape[1]} sütun")
df.to_csv(OUT_DIR / "02_cleaned_data.csv", index=False, encoding="utf-8-sig")
print("\n>>> TABLO: outputs/02_cleaned_data.csv  — temizlenmiş ve doldurulmuş ham veri; sonraki tüm analizler buradan devam eder.")
print(
    "\nVeri temizleme yapıldı mı? Evet. Bu bölümde eksik değerler dolduruldu, yinelenen satırlar kaldırıldı, "
    "mantıksal olarak imkansız değerler NaN yapılıp uygun medyan değerlerle tamamlandı, aykırı değerler IQR yöntemiyle "
    "tespit edilip raporlandı ve kategorik değişkenler analiz/modelleme için sayısal forma dönüştürüldü. "
    "Temizlenmiş veri outputs/02_cleaned_data.csv dosyasına kaydedildi."
)
print(
    "Kardinalite açıklaması: Kardinalite, bir kategorik değişkendeki benzersiz sınıf sayısıdır. "
    "Örneğin gender düşük kardinalitelidir; city ve country daha yüksek kardinalitelidir. "
    "Çok yüksek kardinaliteli değişkenleri doğrudan dummy değişkene çevirmek çok fazla sütun üretir ve analizi gereksiz büyütebilir."
)
print("Yorum: name/city/country gibi çok yüksek kardinaliteli alanlar modelleme dışı tutuldu veya doğrudan encode edilmedi.")


# %% [markdown]
# ## 3. Tanımlayıcı İstatistikler

# %%
# Her sayısal sütun için tanımlayıcı istatistikleri hesapla
print_section("3. TANIMLAYICI İSTATİSTİKLER")

descriptive_rows = []
for col in numeric_cols:
    series = df[col].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    mode_values = series.mode()
    mode_value = mode_values.iloc[0] if not mode_values.empty else np.nan
    skew_value = series.skew()
    kurtosis_value = series.kurtosis()
    descriptive_rows.append(
        {
            "değişken": col,
            "ortalama": series.mean(),
            "medyan": series.median(),
            "mod": mode_value,
            "varyans": series.var(ddof=1),
            "standart_sapma": series.std(ddof=1),
            "çarpıklık": skew_value,
            "çarpıklık_yorumu": skewness_comment(skew_value),
            "basıklık": kurtosis_value,
            "basıklık_yorumu": kurtosis_comment(kurtosis_value),
            "min": series.min(),
            "max": series.max(),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "%25": series.quantile(0.25),
            "%50": series.quantile(0.50),
            "%75": series.quantile(0.75),
        }
    )

descriptive_table = pd.DataFrame(descriptive_rows)
print(descriptive_table)
descriptive_table.to_csv(OUT_DIR / "03_descriptive_statistics.csv", index=False, encoding="utf-8-sig")
print("\n>>> TABLO: outputs/03_descriptive_statistics.csv  — ortalama, medyan, mod, varyans, SS, çarpıklık, basıklık, Q1/Q3/IQR.")

# Tanımlayıcı istatistikleri görselleştir: ortalama, ±1 std bandı, çarpıklık
_desc_cols = [c for c in ["age", "annual_income_usd", "smokes_per_day", "drinks_per_week", "sleep_hours", "bmi"] if c in df.columns]
_ncols = 3
_nrows = int(np.ceil(len(_desc_cols) / _ncols))
fig, axes = plt.subplots(_nrows, _ncols, figsize=(6 * _ncols, 4.5 * _nrows))
axes = np.array(axes).flatten()

for ax, col in zip(axes, _desc_cols):
    series = df[col].dropna()
    mu = series.mean()
    sigma = series.std(ddof=1)
    skew = series.skew()

    sns.histplot(series, kde=True, ax=ax, color="#4c78a8", alpha=0.6, label="Dağılım")
    ax.axvline(mu, color="#e45756", linewidth=2, linestyle="-", label=f"Ort. = {mu:.2f}")
    ax.axvline(series.median(), color="#54a24b", linewidth=1.8, linestyle="--", label=f"Med. = {series.median():.2f}")
    ax.axvspan(mu - sigma, mu + sigma, alpha=0.12, color="#f58518", label=f"±1 SS ({sigma:.2f})")
    skew_label = "sağa çarpık" if skew > 0.5 else ("sola çarpık" if skew < -0.5 else "simetrik")
    ax.text(0.97, 0.95, f"Çarpıklık: {skew:.2f}\n({skew_label})", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.85))
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Frekans")
    ax.legend(fontsize=7.5, loc="upper left")

for ax in axes[len(_desc_cols):]:
    ax.axis("off")

plt.suptitle("Tanımlayıcı İstatistikler — Dağılım, Ortalama, ±1 Std ve Çarpıklık", fontsize=13, y=1.01)
save_current_figure("03_descriptive_overview.png")
print("\n>>> GRAFİK: figures/03_descriptive_overview.png  — her değişken için histogram üzerinde ortalama (kırmızı), medyan (yeşil), ±1 std bandı (turuncu) ve çarpıklık etiketi.")

print(
    "\nÇarpıklık açıklaması: Çarpıklık dağılımın simetrik olup olmadığını gösterir. "
    "Pozitif çarpıklık aynı zamanda sağa çarpıklık demektir; sağ kuyruk uzundur. "
    "Negatif çarpıklık aynı zamanda sola çarpıklık demektir; sol kuyruk uzundur. "
    "0'a yakın değerler yaklaşık simetrik dağılım olarak yorumlanır."
)
print("Beklenen yorum: Ortalama-medyan farkı ve çarpıklık değerleri değişkenlerin dağılım yönü hakkında bilgi verir.")


# %% [markdown]
# ## 4. Görselleştirme

# %%
# Seçili sayısal değişkenler için histogram, box plot ve Q-Q plot üret
print_section("4. GÖRSELLEŞTİRME")
print("Not: Kullanıcının isteğine uygun olarak grafik sayısı sınırlı tutuldu; test bölümlerindeki grafikler de ayrıca üretilecektir.")

for col in visual_numeric_cols:
    series = df[col].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.histplot(series, kde=True, ax=axes[0], color="#4c78a8", label=col)
    axes[0].set_title(f"{col} Histogramı")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Frekans")
    axes[0].legend(loc="best")

    sns.boxplot(y=series, ax=axes[1], color="#f58518")
    axes[1].set_title(f"{col} Box Plot")
    axes[1].set_xlabel("Değişken")
    axes[1].set_ylabel(col)
    axes[1].legend(handles=[plt.Line2D([0], [0], color="#f58518", lw=8, label=col)], loc="best")

    stats.probplot(series, dist="norm", plot=axes[2])
    axes[2].set_title(f"{col} Q-Q Plot")
    axes[2].set_xlabel("Teorik normal kantiller")
    axes[2].set_ylabel("Gözlenen kantiller")
    axes[2].get_lines()[0].set_label("Gözlenen değerler")
    axes[2].get_lines()[1].set_label("Normal referans çizgisi")
    axes[2].legend(loc="best")

    save_current_figure(f"04_numeric_{col}.png")
    print(f"\n>>> GRAFİK: figures/04_numeric_{col}.png  — {col} için Histogram (sol) + Boxplot (orta) + Q-Q Plot (sağ).")


# %%
# Seçili kategorik değişkenler için bar chart/count plot üret
categorical_visual_cols = [
    col
    for col in ["gender", "education_level", "employment_status", "mental_health_status", "has_health_issues", "therapy_history"]
    if col in df.columns
]
categorical_visual_cols = categorical_visual_cols[:4]

if categorical_visual_cols:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, col in zip(axes, categorical_visual_cols):
        counts = df[col].astype(str).value_counts().head(10)
        ax.bar(counts.index, counts.values, color="#54a24b", label="Frekans")
        ax.set_title(f"{col} Dağılımı")
        ax.set_xlabel(col)
        ax.set_ylabel("Frekans")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(loc="best")
    for ax in axes[len(categorical_visual_cols) :]:
        ax.axis("off")
    save_current_figure("04_categorical_countplots.png")
    print("\n>>> GRAFİK: figures/04_categorical_countplots.png  — gender, education_level, employment_status, mental_health_status kategorik dağılımları (4 bar chart).")


# %%
# Korelasyon matrisi heatmap grafiğini çiz
if len(numeric_cols) >= 2:
    print(
        "\nKorelasyon açıklaması: Korelasyon testi iki sayısal değişken arasında doğrusal ilişki olup olmadığını inceler. "
        "Pearson korelasyon katsayısı r -1 ile +1 arasındadır. r pozitifse değişkenler birlikte artma eğilimindedir; "
        "r negatifse biri artarken diğeri azalma eğilimindedir. p-değeri 0.05'ten küçükse ilişki istatistiksel olarak anlamlı kabul edilir."
    )
    corr_matrix = df[numeric_cols].corr(numeric_only=True)
    corr_test_rows = []
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1 :]:
            pair_data = df[[col1, col2]].dropna()
            if len(pair_data) < 3 or pair_data[col1].nunique() < 2 or pair_data[col2].nunique() < 2:
                continue
            r_value, p_value = stats.pearsonr(pair_data[col1], pair_data[col2])
            corr_test_rows.append(
                {
                    "değişken_1": col1,
                    "değişken_2": col2,
                    "pearson_r": r_value,
                    "p_değeri": p_value,
                    "yorum": "Anlamlı doğrusal ilişki var." if p_value < 0.05 else "Anlamlı doğrusal ilişki yok.",
                }
            )
    corr_test_table = pd.DataFrame(corr_test_rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
    print("\n# Pearson korelasyon testi özeti")
    print(corr_test_table.head(15).to_string(index=False))
    corr_test_table.to_csv(OUT_DIR / "04_correlation_tests.csv", index=False, encoding="utf-8-sig")
    print("\n>>> TABLO: outputs/04_correlation_tests.csv  — tüm değişken çiftleri için Pearson r ve p-değeri; en güçlü korelasyonlar üstte.")

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Korelasyon katsayısı"},
    )
    plt.title("Sayısal Değişkenler Korelasyon Matrisi")
    plt.xlabel("Değişkenler")
    plt.ylabel("Değişkenler")
    save_current_figure("04_correlation_heatmap.png")


print("\nBeklenen yorum: Histogramlar dağılımı, box plotlar aykırı değerleri, Q-Q plotlar normalliği görsel olarak kontrol eder.")


# %% [markdown]
# ## 5. Normallik Testleri

# %%
# Her sayısal sütun için Shapiro-Wilk veya D'Agostino K2 ve Anderson-Darling testlerini uygula
print_section("5. NORMALLİK TESTLERİ")

normality_rows = []
alpha = 0.05

for col in numeric_cols:
    values = df[col].dropna().astype(float)
    n = len(values)

    if n < 3 or values.nunique() < 2:
        normality_rows.append(
            {
                "değişken": col,
                "n": n,
                "ana_test": "Uygulanamadı",
                "test_istatistiği": np.nan,
                "p_değeri": np.nan,
                "p_kararı": "Yetersiz veri veya sabit değişken",
                "anderson_istatistiği": np.nan,
                "anderson_5pct_kritik": np.nan,
                "anderson_kararı": "Uygulanamadı",
            }
        )
        continue

    if n <= 50:
        test_name = "Shapiro-Wilk"
        test_stat, p_value = stats.shapiro(values)
    else:
        test_name = "D'Agostino K2"
        test_stat, p_value = stats.normaltest(values)

    anderson_result = stats.anderson(values, dist="norm")
    significance_levels = np.array(anderson_result.significance_level)
    idx_5 = int(np.argmin(np.abs(significance_levels - 5)))
    ad_critical = float(anderson_result.critical_values[idx_5])
    ad_stat = float(anderson_result.statistic)
    ad_decision = (
        "H0 reddedilir: normal dağılıma uygun değildir."
        if ad_stat > ad_critical
        else "H0 reddedilemez: normal dağılıma aykırı güçlü kanıt yoktur."
    )

    normality_rows.append(
        {
            "değişken": col,
            "n": n,
            "ana_test": test_name,
            "test_istatistiği": test_stat,
            "p_değeri": p_value,
            "p_kararı": normality_decision(p_value, alpha),
            "anderson_istatistiği": ad_stat,
            "anderson_5pct_kritik": ad_critical,
            "anderson_kararı": ad_decision,
        }
    )

normality_table = pd.DataFrame(normality_rows)
print("H0: Değişken normal dağılıma uygundur.")
print("H1: Değişken normal dağılıma uygun değildir.")
print(normality_table)
normality_table.to_csv(OUT_DIR / "05_normality_tests.csv", index=False, encoding="utf-8-sig")
print("\nYorum: p < 0.05 ise H0 reddedilir ve değişkenin normal dağılmadığı sonucuna varılır.")

# D'Agostino K2 sonuçlarını görselleştir: histogram + normal eğri + karar etiketi
_dagostino_rows = normality_table[normality_table["ana_test"] == "D'Agostino K2"].reset_index(drop=True)
if not _dagostino_rows.empty:
    _d_cols = _dagostino_rows["değişken"].tolist()
    _d_ncols = 3
    _d_nrows = int(np.ceil(len(_d_cols) / _d_ncols))
    fig, axes = plt.subplots(_d_nrows, _d_ncols, figsize=(6 * _d_ncols, 4.5 * _d_nrows))
    axes = np.array(axes).flatten()

    for ax, (_, row) in zip(axes, _dagostino_rows.iterrows()):
        col = row["değişken"]
        series = df[col].dropna().astype(float)
        mu, sigma = series.mean(), series.std(ddof=1)
        x_range = np.linspace(series.min(), series.max(), 300)

        sns.histplot(series, kde=False, stat="density", ax=ax, color="#4c78a8", alpha=0.55, label="Gözlenen")
        ax.plot(x_range, stats.norm.pdf(x_range, loc=mu, scale=sigma), color="#e45756", linewidth=2, label="Normal PDF")

        rejected = (not pd.isna(row["p_değeri"])) and (row["p_değeri"] < alpha)
        karar_text = f"K²={row['test_istatistiği']:.2f}\np={row['p_değeri']:.4f}\n{'H0 REDDEDİLİR ✗' if rejected else 'H0 REDDEDİLEMEZ ✓'}"
        box_color = "#ffe0e0" if rejected else "#e0ffe0"
        ax.text(0.97, 0.95, karar_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=box_color, edgecolor="#aaaaaa", alpha=0.9))
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Yoğunluk")
        ax.legend(fontsize=7.5, loc="upper left")

    for ax in axes[len(_d_cols):]:
        ax.axis("off")

    plt.suptitle("D'Agostino K² Normallik Testi — Histogram vs Normal Dağılım", fontsize=13, y=1.01)
    save_current_figure("05_dagostino_normality.png")
    print("\n>>> GRAFİK: figures/05_dagostino_normality.png  — her değişken için gözlenen dağılım (mavi) ile normal eğri (kırmızı) karşılaştırması; kutucukta K² istatistiği, p-değeri ve H0 kararı.")


# %%
# Küçük örneklem ve alt gruplar için Shapiro-Wilk normallik testi uygula
print("\n# Ek Shapiro-Wilk testi: küçük örneklem alt grup normallik kontrolü")
print(
    "Neden Shapiro-Wilk? Shapiro-Wilk testi küçük örneklemlerde normalliği kontrol etmek için uygundur. "
    "Bu nedenle tüm veri yerine alt gruplardan n=30 gözlem alınarak uygulanmıştır."
)
print("H0: Alt gruptaki sleep_hours örneklemi normal dağılıma uygundur.")
print("H1: Alt gruptaki sleep_hours örneklemi normal dağılıma uygun değildir.")

shapiro_rows = []
shapiro_group_col = "gender"
shapiro_value_col = "sleep_hours" if "sleep_hours" in df.columns else numeric_cols[0]
shapiro_groups = ["Male", "Female"] if shapiro_group_col in df.columns else []

if shapiro_group_col in df.columns and shapiro_value_col in df.columns:
    fig, axes = plt.subplots(len(shapiro_groups), 2, figsize=(11, 4.4 * len(shapiro_groups)))
    if len(shapiro_groups) == 1:
        axes = np.array([axes])

    for row_idx, group_name in enumerate(shapiro_groups):
        group_series = df.loc[df[shapiro_group_col] == group_name, shapiro_value_col].dropna().astype(float)
        sample_size = min(30, len(group_series))
        sample_series = group_series.sample(n=sample_size, random_state=42) if sample_size > 0 else group_series

        if sample_size >= 3 and sample_series.nunique() >= 2:
            shapiro_stat, shapiro_p = stats.shapiro(sample_series)
            shapiro_comment = normality_decision(shapiro_p, alpha)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
            shapiro_comment = "Test uygulanamadı: yeterli veri yok."

        shapiro_rows.append(
            {
                "grup_değişkeni": shapiro_group_col,
                "grup": group_name,
                "test_değişkeni": shapiro_value_col,
                "örneklem_boyutu": sample_size,
                "shapiro_istatistiği": shapiro_stat,
                "p_değeri": shapiro_p,
                "karar": shapiro_comment,
            }
        )

        sns.histplot(sample_series, kde=True, ax=axes[row_idx, 0], color="#4c78a8", label=f"{group_name} örneklem")
        axes[row_idx, 0].set_title(f"{group_name} - {shapiro_value_col} Histogramı (n={sample_size})")
        axes[row_idx, 0].set_xlabel(shapiro_value_col)
        axes[row_idx, 0].set_ylabel("Frekans")
        axes[row_idx, 0].legend(loc="best")

        stats.probplot(sample_series, dist="norm", plot=axes[row_idx, 1])
        axes[row_idx, 1].set_title(f"{group_name} - {shapiro_value_col} Q-Q Plot")
        axes[row_idx, 1].set_xlabel("Teorik normal kantiller")
        axes[row_idx, 1].set_ylabel("Gözlenen kantiller")
        axes[row_idx, 1].get_lines()[0].set_label("Gözlenen değerler")
        axes[row_idx, 1].get_lines()[1].set_label("Normal referans çizgisi")
        axes[row_idx, 1].legend(loc="best")

    save_current_figure("05_shapiro_small_group_sleep_hours.png")
else:
    print("Shapiro-Wilk alt grup testi uygulanamadı: gerekli sütunlar bulunamadı.")

shapiro_group_table = pd.DataFrame(shapiro_rows)
print(shapiro_group_table.to_string(index=False))
shapiro_group_table.to_csv(OUT_DIR / "05_shapiro_small_group_tests.csv", index=False, encoding="utf-8-sig")
print("Yorum: Bu ek test, küçük örneklemde alt grupların normal dağılıma uyup uymadığını göstermek için kullanıldı.")


# %% [markdown]
# ## 6. Olasılık Dağılımları

# %%
# Bir sürekli değişken için normal dağılım PDF grafiği çiz
print_section("6. OLASILIK DAĞILIMLARI")

normal_col = "sleep_hours" if "sleep_hours" in df.columns else numeric_cols[0]
normal_series = df[normal_col].dropna().astype(float)
normal_mu = normal_series.mean()
normal_sigma = normal_series.std(ddof=1)
x_values = np.linspace(normal_series.min(), normal_series.max(), 300)

plt.figure(figsize=(9, 5))
plt.hist(normal_series, bins=30, density=True, alpha=0.45, color="#4c78a8", label=f"Gözlenen {normal_col}")
plt.plot(x_values, stats.norm.pdf(x_values, loc=normal_mu, scale=normal_sigma), color="#e45756", linewidth=2, label="Normal PDF")
plt.title(f"{normal_col} İçin Normal Dağılım Yaklaşımı")
plt.xlabel(normal_col)
plt.ylabel("Yoğunluk")
plt.legend(loc="best")
save_current_figure(f"06_normal_pdf_{normal_col}.png")
print(f"{normal_col} için normal dağılım parametreleri: ortalama={normal_mu:.4f}, std={normal_sigma:.4f}")


print(
    "Yorum: Bu bölümde yalnızca sürekli bir sayısal değişken için normal dağılım yaklaşımı gösterildi. "
    
)


# %% [markdown]
# ## 7. Örnekleme ve Merkezi Limit Teoremi

# %%
# Basit rastgele örneklem ortalamalarıyla Merkezi Limit Teoremi'ni göster
print_section("7. ÖRNEKLEME VE MERKEZİ LİMİT TEOREMİ")

clt_col = normal_col
clt_values = df[clt_col].dropna().astype(float).to_numpy()
population_mean = float(np.mean(clt_values))
population_std = float(np.std(clt_values, ddof=1))
rng = np.random.default_rng(42)
sample_sizes = [30, 50, 100]
clt_rows = []

fig, axes = plt.subplots(1, len(sample_sizes), figsize=(16, 4.5))

for ax, sample_size in zip(axes, sample_sizes):
    means = np.array([rng.choice(clt_values, size=sample_size, replace=False).mean() for _ in range(1000)])
    theoretical_se = population_std / np.sqrt(sample_size)
    empirical_se = means.std(ddof=1)
    clt_rows.append(
        {
            "değişken": clt_col,
            "örneklem_boyutu": sample_size,
            "ana_kütle_ortalaması": population_mean,
            "ana_kütle_std": population_std,
            "teorik_SE": theoretical_se,
            "örneklem_ortalamaları_std": empirical_se,
        }
    )
    sns.histplot(means, kde=True, ax=ax, color="#4c78a8", label=f"n={sample_size}")
    ax.axvline(population_mean, color="#e45756", linestyle="--", linewidth=2, label="Ana kütle ortalaması")
    ax.set_title(f"Örneklem Ortalamaları (n={sample_size})")
    ax.set_xlabel(f"{clt_col} örneklem ortalaması")
    ax.set_ylabel("Frekans")
    ax.legend(loc="best")

save_current_figure(f"07_clt_{clt_col}.png")

clt_table = pd.DataFrame(clt_rows)
print(clt_table)
clt_table.to_csv(OUT_DIR / "07_clt_summary.csv", index=False, encoding="utf-8-sig")
print("Yorum: Örneklem boyutu arttıkça ortalamaların dağılımı daralır; SE = standart sapma / karekok(n) formülü bunu açıklar.")


# %% [markdown]
# ## 8. Güven Aralığı

# %%
# En az iki sayısal değişken için %90, %95 ve %99 güven aralıklarını hesapla
print_section("8. GÜVEN ARALIĞI")

ci_cols = [col for col in ["sleep_hours", "bmi"] if col in df.columns]
if len(ci_cols) < 2:
    ci_cols = numeric_cols[:2]

z_values = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
ci_rows = []

for col in ci_cols:
    series = df[col].dropna().astype(float)
    n = len(series)
    mean_value = series.mean()
    std_value = series.std(ddof=1)
    se_value = std_value / np.sqrt(n)

    for confidence, z_critical in z_values.items():
        lower = mean_value - z_critical * se_value
        upper = mean_value + z_critical * se_value
        scipy_lower, scipy_upper = safe_norm_interval(confidence, loc=mean_value, scale=se_value)
        ci_rows.append(
            {
                "değişken": col,
                "güven_düzeyi": f"%{int(confidence * 100)}",
                "n": n,
                "ortalama": mean_value,
                "std": std_value,
                "SE": se_value,
                "Z_kritik": z_critical,
                "formül_alt": lower,
                "formül_üst": upper,
                "scipy_alt": scipy_lower,
                "scipy_üst": scipy_upper,
            }
        )

ci_table = pd.DataFrame(ci_rows)
print(ci_table)
ci_table.to_csv(OUT_DIR / "08_confidence_intervals.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, len(ci_cols), figsize=(6 * len(ci_cols), 4.8))
if len(ci_cols) == 1:
    axes = [axes]

for ax, col in zip(axes, ci_cols):
    plot_data = ci_table[ci_table["değişken"] == col].copy()
    y_positions = np.arange(len(plot_data))
    means = plot_data["ortalama"].to_numpy()
    lower_errors = means - plot_data["formül_alt"].to_numpy()
    upper_errors = plot_data["formül_üst"].to_numpy() - means

    ax.errorbar(
        means,
        y_positions,
        xerr=[lower_errors, upper_errors],
        fmt="o",
        color="#4c78a8",
        ecolor="#f58518",
        elinewidth=2,
        capsize=5,
        label="Ortalama ve güven aralığı",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["güven_düzeyi"])
    ax.set_title(f"{col} Güven Aralıkları")
    ax.set_xlabel(col)
    ax.set_ylabel("Güven düzeyi")
    ax.legend(loc="best")

save_current_figure("08_confidence_intervals.png")
print("Grafik yorumu: Noktalar örneklem ortalamasını, yatay çizgiler güven aralığının alt ve üst sınırını gösterir.")
print("Yorum: Güven düzeyi arttıkça Z kritik değeri büyür ve güven aralığı genişler.")


# %% [markdown]
# ## 9. Hipotez Testleri

# %%
# Hipotez 1: Tek örneklem t testi
print_section("9. HİPOTEZ TESTLERİ")

hypothesis_rows = []
alpha = 0.05

print("\nHipotez 1 - Tek örneklem t testi")
h1_col = "sleep_hours" if "sleep_hours" in df.columns else numeric_cols[0]
h1_reference = 7.0
h1_series = df[h1_col].dropna().astype(float)
h1_stat, h1_p = stats.ttest_1samp(h1_series, popmean=h1_reference)
h1_decision = hypothesis_decision(h1_p, alpha)
h1_mean = h1_series.mean()
h1_mean_diff = h1_mean - h1_reference

print("-" * 60)
print("  H0 : Popülasyondaki ortalama uyku süresi 7 saattir.  (μ = 7)")
print("  H1 : Popülasyondaki ortalama uyku süresi 7 saat değildir.  (μ ≠ 7)")
print("-" * 60)
print("Adım 1: H0: mu_sleep_hours = 7, H1: mu_sleep_hours != 7")
print("Adım 2: Anlamlılık düzeyi alpha = 0.05")
print(
    "Adım 3: Tek örneklem t testi seçildi; çünkü tek bir sayısal değişkenin ortalamasını sabit bir referans değerle "
    "karşılaştırıyoruz. Ana kütle standart sapması bilinmediği için Z testi yerine t testi daha uygundur. "
    "Ayrıca n=3000 olduğu için Merkezi Limit Teoremi ortalamanın örnekleme dağılımını destekler."
)
print(f"Adım 4: t istatistiği = {h1_stat:.6f}")
print(
    f"Adım 5: p-değeri = {h1_p:.6g}; karar = {h1_decision}. "
    "p-değeri 0.05'ten küçükse H0 reddedilir; 0.05 veya daha büyükse H0 reddedilemez."
)
print(
    "Adım 6: "
    + (
        f"Örneklem ortalaması {h1_mean:.2f} saattir; bu değer 7 saatten {abs(h1_mean_diff):.2f} saat "
        f"{'daha düşüktür' if h1_mean_diff < 0 else 'daha yüksektir'}. "
        "p-değeri 0.05'ten küçük olduğu için bu fark istatistiksel olarak anlamlıdır. "
        "Yani veri setindeki ortalama uyku süresinin 7 saat olduğu varsayımı desteklenmemektedir."
        if h1_p < alpha
        else f"Örneklem ortalaması {h1_mean:.2f} saattir; 7 saatten görünen fark {abs(h1_mean_diff):.2f} saattir. "
        "Ancak p-değeri 0.05'ten büyük/eşit olduğu için bu farkın tesadüfi örneklem dalgalanmasından kaynaklanmadığını "
        "söylemek için yeterli kanıt yoktur. Bu sonuç 'ortalama kesinlikle 7 saattir' demek değildir; sadece elimizdeki "
        "veriyle 7 saatten anlamlı biçimde farklıdır diyemiyoruz."
    )
)

hypothesis_rows.append(
    {
        "hipotez": "H1",
        "test": "Tek örneklem t testi",
        "değişkenler": f"{h1_col} vs 7 saat",
        "istatistik": h1_stat,
        "p_değeri": h1_p,
        "karar": h1_decision,
    }
)

plt.figure(figsize=(9, 5))
sns.histplot(h1_series, kde=True, color="#4c78a8", label=h1_col)
plt.axvline(h1_reference, color="#e45756", linestyle="--", linewidth=2, label="H0 ortalaması: 7")
plt.axvline(h1_mean, color="#54a24b", linestyle="-", linewidth=2, label=f"Örneklem ortalaması: {h1_mean:.2f}")
plt.title("Hipotez 1: Uyku Süresi Ortalaması")
plt.xlabel(h1_col)
plt.ylabel("Frekans")
plt.legend(loc="best")
save_current_figure("09_h1_sleep_hours_ttest.png")


# %%
# Hipotez 2: İki bağımsız grup için Mann-Whitney U testi
print("\nHipotez 2 - İki bağımsız grup Mann-Whitney U testi")
h2_group_col = "has_health_issues"
h2_value_col = "smokes_per_day" if "smokes_per_day" in df.columns else numeric_cols[0]


def health_issue_label(value) -> str:
    """Grafik ve yorumlarda True/False değerlerini anlaşılır sağlık sorunu etiketlerine çevirir."""
    value_text = str(value).lower()
    if value_text == "false":
        return "False = sağlık sorunu yok"
    if value_text == "true":
        return "True = sağlık sorunu var"
    return str(value)


if h2_group_col in df.columns:
    group_values = list(df[h2_group_col].dropna().unique())
    if len(group_values) >= 2:
        group_a, group_b = group_values[:2]
        group_a_label = health_issue_label(group_a)
        group_b_label = health_issue_label(group_b)
        h2_a = df.loc[df[h2_group_col] == group_a, h2_value_col].dropna().astype(float)
        h2_b = df.loc[df[h2_group_col] == group_b, h2_value_col].dropna().astype(float)
        h2_stat, h2_p = stats.mannwhitneyu(h2_a, h2_b, alternative="two-sided")
        h2_decision = hypothesis_decision(h2_p, alpha)
        h2_group_summary = (
            df.groupby(h2_group_col)[h2_value_col]
            .agg(["count", "median", "mean", "min", "max"])
            .rename(
                columns={
                    "count": "kişi_sayısı",
                    "median": "medyan",
                    "mean": "ortalama",
                    "min": "minimum",
                    "max": "maksimum",
                }
            )
        )
        h2_group_summary.to_csv(OUT_DIR / "09_h2_group_summary.csv", encoding="utf-8-sig")

        print("-" * 60)
        print(f"  H0 : Sağlık sorunu olan ve olmayan bireylerin günlük sigara sayısı dağılımı aynıdır.")
        print(f"  H1 : Sağlık sorunu olan ve olmayan bireylerin günlük sigara sayısı dağılımı farklıdır.")
        print("-" * 60)
        print(f"Adım 1: H0: {h2_value_col} dağılımı {group_a_label} ve {group_b_label} gruplarında aynıdır.")
        print(f"        H1: {h2_value_col} dağılımı {group_a_label} ve {group_b_label} gruplarında farklıdır.")
        print("Adım 2: Anlamlılık düzeyi alpha = 0.05")
        print(
            "Adım 3: Değişken sayım tipinde olduğundan ve normal dağılım varsayımı güçlü olmadığından Mann-Whitney U testi seçildi. "
            "Bu test iki bağımsız grubun dağılımlarının/merkezi eğilimlerinin belirgin biçimde farklı olup olmadığını kontrol eder."
        )
        print(f"Adım 4: U istatistiği = {h2_stat:.6f}")
        print(
            f"Adım 5: p-değeri = {h2_p:.6g}; karar = {h2_decision}. "
            "p-değeri 0.05'ten büyük/eşit olduğunda H0 reddedilemez; yani gruplar arasında fark var demek için yeterli kanıt yoktur."
        )
        print("\nHipotez 2 grup özeti")
        print(h2_group_summary)
        print(
            "Adım 6: "
            + (
                f"{group_a_label} ve {group_b_label} grupları arasında {h2_value_col} açısından istatistiksel olarak anlamlı fark vardır. "
                "Bu, sağlık sorunu durumuna göre günlük sigara sayısı dağılımlarının aynı kabul edilemeyeceği anlamına gelir."
                if h2_p < alpha
                else f"{group_a_label} ve {group_b_label} grupları arasında {h2_value_col} açısından istatistiksel olarak anlamlı fark bulunmamıştır. "
                "Bunu 'iki grup kesinlikle aynıdır' diye okumamalıyız; sadece bu veri ve alpha=0.05 düzeyinde, "
                "sağlık sorunu olanlarla olmayanların günlük sigara sayılarının farklı olduğunu gösterecek kadar güçlü kanıt yoktur. "
                "Grafikteki kutuların ve medyan çizgilerinin birbirine çok yakın görünmesi de bu sonucu görsel olarak destekler."
            )
        )

        hypothesis_rows.append(
            {
                "hipotez": "H2",
                "test": "Mann-Whitney U",
                "değişkenler": f"{h2_group_col} gruplarına göre {h2_value_col}",
                "istatistik": h2_stat,
                "p_değeri": h2_p,
                "karar": h2_decision,
            }
        )

        plt.figure(figsize=(8, 5))
        h2_plot_data = df[[h2_group_col, h2_value_col]].copy()
        h2_plot_data["sağlık_sorunu_durumu"] = h2_plot_data[h2_group_col].apply(health_issue_label)
        sns.boxplot(
            x="sağlık_sorunu_durumu",
            y=h2_value_col,
            data=h2_plot_data,
            order=[health_issue_label(value) for value in group_values[:2]],
            color="#9ecae1",
        )
        plt.title("Hipotez 2: Sağlık Sorunu Durumuna Göre Günlük Sigara Sayısı")
        plt.xlabel("Sağlık sorunu durumu")
        plt.ylabel("Günlük sigara sayısı")
        save_current_figure("09_h2_mannwhitney_smoking_health.png")
    else:
        print("Hipotez 2 uygulanamadı: grup değişkeninde en az iki grup yok.")
else:
    print("Hipotez 2 uygulanamadı: has_health_issues sütunu bulunamadı.")


# %%
# Hipotez 3: İki örneklem oran Z testi
print("\nHipotez 3 - İki örneklem oran Z testi")
h3_group_col = "gender"
h3_binary_col = "has_health_issues"

if {h3_group_col, h3_binary_col}.issubset(df.columns):
    h3_data = df[df[h3_group_col].isin(["Male", "Female"])].copy()
    h3_data["health_binary"] = bool_series_to_int(h3_data[h3_binary_col])
    h3_data = h3_data.dropna(subset=["health_binary"])

    group_order = ["Male", "Female"]
    successes = h3_data.groupby(h3_group_col)["health_binary"].sum().reindex(group_order).astype(float)
    nobs = h3_data.groupby(h3_group_col)["health_binary"].count().reindex(group_order).astype(float)

    if successes.notna().all() and nobs.notna().all() and (nobs > 0).all():
        h3_stat, h3_p = proportions_ztest(count=successes.to_numpy(), nobs=nobs.to_numpy(), alternative="two-sided")
        h3_decision = hypothesis_decision(h3_p, alpha)
        h3_props = successes / nobs

        print("-" * 60)
        print("  H0 : Erkek ve kadınlarda sağlık sorunu görülme oranı eşittir.  (p_Male = p_Female)")
        print("  H1 : Erkek ve kadınlarda sağlık sorunu görülme oranı farklıdır.  (p_Male ≠ p_Female)")
        print("-" * 60)
        print("Adım 1: H0: p_Male = p_Female, H1: p_Male != p_Female")
        print("Adım 2: Anlamlılık düzeyi alpha = 0.05")
        print("Adım 3: İki bağımsız grubun sağlık sorunu oranları karşılaştırıldığı için iki örneklem oran Z testi seçildi.")
        print(f"Adım 4: Z istatistiği = {h3_stat:.6f}")
        print(f"Adım 5: p-değeri = {h3_p:.6g}; karar = {h3_decision}")
        print(
            "Adım 6: "
            + (
                "Male ve Female gruplarının sağlık sorunu oranları arasında anlamlı fark vardır."
                if h3_p < alpha
                else "Male ve Female gruplarının sağlık sorunu oranları arasında anlamlı fark bulunmamıştır."
            )
        )

        hypothesis_rows.append(
            {
                "hipotez": "H3",
                "test": "İki örneklem oran Z testi",
                "değişkenler": "gender gruplarında has_health_issues oranı",
                "istatistik": h3_stat,
                "p_değeri": h3_p,
                "karar": h3_decision,
            }
        )

        ci_error = 1.96 * np.sqrt(h3_props * (1 - h3_props) / nobs)
        plt.figure(figsize=(8, 5))
        plt.bar(group_order, h3_props.loc[group_order], yerr=ci_error.loc[group_order], capsize=6, color="#72b7b2", label="Sağlık sorunu oranı")
        plt.title("Hipotez 3: Cinsiyete Göre Sağlık Sorunu Oranı")
        plt.xlabel("Cinsiyet")
        plt.ylabel("Oran")
        plt.ylim(0, min(1, float((h3_props + ci_error).max() + 0.1)))
        plt.legend(loc="best")
        save_current_figure("09_h3_two_proportion_gender_health.png")
    else:
        print("Hipotez 3 uygulanamadı: Male/Female gruplarında yeterli veri yok.")
else:
    print("Hipotez 3 uygulanamadı: gender veya has_health_issues sütunu bulunamadı.")


# %%
# Hipotez testleri özet tablosunu yazdır
hypothesis_table = pd.DataFrame(hypothesis_rows)
print("\n# Hipotez testleri özet tablosu")
print(hypothesis_table)
hypothesis_table.to_csv(OUT_DIR / "09_hypothesis_tests.csv", index=False, encoding="utf-8-sig")


# %% [markdown]
# ## 10. Sonuç ve Yorum

# %%
# Tüm bulguları özetle ve yorumla
print_section("10. SONUÇ VE YORUM")

normal_vars = normality_table[
    (normality_table["p_değeri"] >= alpha)
    & (normality_table["anderson_kararı"].str.startswith("H0 reddedilemez", na=False))
]["değişken"].tolist()

non_normal_vars = normality_table[
    (normality_table["p_değeri"] < alpha)
    | (normality_table["anderson_kararı"].str.startswith("H0 reddedilir", na=False))
]["değişken"].tolist()

print("# Normal dağılıma uygun görünen değişkenler")
print(normal_vars if normal_vars else "Bu kriterlerle normal dağılıma uygun değişken bulunmadı.")

print("\n# Normal dağılmayan değişkenler")
print(non_normal_vars if non_normal_vars else "Bu kriterlerle normal dışı değişken bulunmadı.")

strong_corr_rows = []
if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr(numeric_only=True)
    for i, col1 in enumerate(corr_matrix.columns):
        for col2 in corr_matrix.columns[i + 1 :]:
            corr_value = corr_matrix.loc[col1, col2]
            if abs(corr_value) >= 0.50:
                strong_corr_rows.append(
                    {
                        "değişken_1": col1,
                        "değişken_2": col2,
                        "korelasyon": corr_value,
                        "yorum": "Pozitif güçlü/orta ilişki" if corr_value > 0 else "Negatif güçlü/orta ilişki",
                    }
                )

strong_corr_table = pd.DataFrame(strong_corr_rows)
print("\n# |r| >= 0.50 olan korelasyonlar")
print(strong_corr_table if not strong_corr_table.empty else "0.50 ve üzeri güçlü/orta korelasyon bulunmadı.")
strong_corr_table.to_csv(OUT_DIR / "10_strong_correlations.csv", index=False, encoding="utf-8-sig")

summary_table = pd.DataFrame(
    [
        {"başlık": "Veri boyutu", "bulgu": f"{df.shape[0]} satır x {df.shape[1]} sütun"},
        {"başlık": "Başlangıç eksik değer sayısı", "bulgu": int(missing_table["eksik_sayı"].sum())},
        {"başlık": "Yinelenen satır sayısı", "bulgu": duplicate_count},
        {"başlık": "En çok aykırı değer içeren değişken", "bulgu": outlier_table.iloc[0]["değişken"] if not outlier_table.empty else "Yok"},
        {"başlık": "Normal görünen değişken sayısı", "bulgu": len(normal_vars)},
        {"başlık": "Normal görünmeyen değişken sayısı", "bulgu": len(non_normal_vars)},
        {"başlık": "Hipotez testi sayısı", "bulgu": len(hypothesis_table)},
    ]
)

print("\n# Genel bulgular özet tablosu")
print(summary_table)
summary_table.to_csv(OUT_DIR / "10_summary.csv", index=False, encoding="utf-8-sig")

print("\n# Hipotez testlerinin genel bulguları")
if not hypothesis_table.empty:
    for _, row in hypothesis_table.iterrows():
        print(f"{row['hipotez']} - {row['test']}: p={row['p_değeri']:.6g}, karar={row['karar']}")
else:
    print("Hipotez testi üretilemedi.")

print("\n# Veri setinin kullanılabileceği amaçlar")
print(
    "- Bağımlılık davranışları ile sağlık, uyku, egzersiz ve mental sağlık göstergeleri arasındaki ilişkileri incelemek.\n"
    "- Farklı demografik gruplarda sigara/alkol kullanımı ve bırakma denemelerini karşılaştırmak.\n"
    "- Risk profili çıkarma, halk sağlığı farkındalık analizi ve istatistiksel modelleme çalışmaları yapmak."
)

print("\nSon yorum: Grafikler 'figures' klasörüne, tablolar 'outputs' klasörüne kaydedildi.")
