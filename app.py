import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="Restaurant Dashboard",
    page_icon="🍽",
    layout="wide"
)

# --------------------------------------------
# LOAD DATASET
# --------------------------------------------

dataset = pd.read_csv("Dataset.csv")

# -------------------------------
# Data Preprocessing
# -------------------------------

dataset["Cuisines"] = dataset["Cuisines"].fillna(
    dataset["Cuisines"].mode()[0]
)

dataset["City"] = dataset["City"].fillna(
    dataset["City"].mode()[0]
)

dataset["Average Cost for two"] = dataset["Average Cost for two"].fillna(
    dataset["Average Cost for two"].median()
)

dataset["Aggregate rating"] = dataset["Aggregate rating"].fillna(
    dataset["Aggregate rating"].median()
)

dataset["Has Table booking"] = dataset["Has Table booking"].apply(
    lambda x: 1 if x=="Yes" else 0
)

dataset["Has Online delivery"] = dataset["Has Online delivery"].apply(
    lambda x: 1 if x=="Yes" else 0
)

city_encoder = LabelEncoder()

dataset["City_encoded"] = city_encoder.fit_transform(
    dataset["City"]
)

tfidf_vectorizer = TfidfVectorizer()

cuisine_matrix = tfidf_vectorizer.fit_transform(
    dataset["Cuisines"]
)

features = pd.concat(
    [
        pd.DataFrame(cuisine_matrix.toarray()),
        dataset[
            [
                "Average Cost for two",
                "Aggregate rating",
                "Has Table booking",
                "Has Online delivery",
                "City_encoded",
            ]
        ],
    ],
    axis=1,
)

features.columns = features.columns.astype(str)

knn_model = NearestNeighbors(
    n_neighbors=5,
    algorithm="auto"
)

knn_model.fit(features)

def recommend_restaurants(
    cuisine,
    cost,
    rating,
    city,
    table_booking=False,
    online_delivery=False,
    n_recommendations=5,
):

    if city not in city_encoder.classes_:

        city_encoded = city_encoder.transform(
            [dataset["City"].mode()[0]]
        )[0]

    else:

        city_encoded = city_encoder.transform([city])[0]

    user_cuisine = tfidf_vectorizer.transform(
        [cuisine]
    ).toarray()

    table = 1 if table_booking else 0
    delivery = 1 if online_delivery else 0

    user = np.hstack(
        (
            user_cuisine[0],
            [
                cost,
                rating,
                table,
                delivery,
                city_encoded,
            ],
        )
    ).reshape(1, -1)

    user_df = pd.DataFrame(
        user,
        columns=features.columns,
    )

    distances, indices = knn_model.kneighbors(
        user_df,
        n_neighbors=n_recommendations,
    )

    recommendations = dataset.iloc[indices[0]].copy()

    recommendations["Has Table booking"] = recommendations[
        "Has Table booking"
    ].replace(
        {0: "No", 1: "Yes"}
    )

    recommendations["Has Online delivery"] = recommendations[
        "Has Online delivery"
    ].replace(
        {0: "No", 1: "Yes"}
    )

    return recommendations[
        [
            "Restaurant Name",
            "Cuisines",
            "City",
            "Address",
            "Average Cost for two",
            "Aggregate rating",
            "Has Table booking",
            "Has Online delivery",
        ]
    ]

# --------------------------------------------
# CUSTOM CSS
# --------------------------------------------

st.markdown("""
<style>

.stApp{
    background:#f7f8fc;
}

/* Hide Streamlit menu */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

.hero{
    background:linear-gradient(135deg,#0f172a,#1e293b,#b91c1c);
    padding:35px;
    border-radius:18px;
    color:white;
    box-shadow:0 8px 20px rgba(0,0,0,.15);
}

.hero h1{
    margin:0;
    font-size:42px;
}

.hero p{
    margin-top:10px;
    color:#d1d5db;
    font-size:17px;
}

.kpi{
    background:white;
    padding:20px;
    border-radius:15px;
    border:1px solid #ececec;
    box-shadow:0 3px 10px rgba(0,0,0,.08);
}

.kpi-title{
    color:#6b7280;
    font-size:15px;
}

.kpi-value{
    color:#b91c1c;
    font-size:34px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# SIDEBAR
# --------------------------------------------

with st.sidebar:

    st.markdown("## 📋 Project Summary")

    st.write("**Dataset:** Restaurant Dataset")

    st.write("**Model:** K-Nearest Neighbors")

    st.write("**Technique:** TF-IDF + KNN")

    st.write("**Restaurants:** {:,}".format(len(dataset)))

    st.divider()

    st.markdown("### 💡 Viva Points")

    st.info("""

• Uses TF-IDF Vectorization

• Uses Label Encoding

• KNN finds similar restaurants

• Recommendation based on:

    ✓ Cuisine

    ✓ City

    ✓ Budget

    ✓ Rating

""")

# --------------------------------------------
# HERO
# --------------------------------------------

st.markdown("""

<div class="hero">

<h1>🍽 Restaurant Dashboard</h1>

<p>

AI Powered Recommendation System using

K-Nearest Neighbors (KNN)

</p>

</div>

""", unsafe_allow_html=True)

st.write("")

# --------------------------------------------
# KPI CARDS
# --------------------------------------------

c1,c2,c3,c4=st.columns(4)

with c1:

    st.markdown(f"""

<div class="kpi">

<div class="kpi-title">Restaurants</div>

<div class="kpi-value">{len(dataset):,}</div>

</div>

""",unsafe_allow_html=True)

with c2:

    st.markdown(f"""

<div class="kpi">

<div class="kpi-title">Cities</div>

<div class="kpi-value">{dataset['City'].nunique()}</div>

</div>

""",unsafe_allow_html=True)

with c3:

    st.markdown(f"""

<div class="kpi">

<div class="kpi-title">Cuisine Types</div>

<div class="kpi-value">{dataset['Cuisines'].nunique()}</div>

</div>

""",unsafe_allow_html=True)

with c4:

    st.markdown(f"""

<div class="kpi">

<div class="kpi-title">Average Rating</div>

<div class="kpi-value">{round(dataset['Aggregate rating'].mean(),2)}</div>

</div>

""",unsafe_allow_html=True)

st.write("")

# --------------------------------------------
# TABS
# --------------------------------------------

tab1,tab2=st.tabs([
    "🍽 Recommendation",
    "📊 Dataset Analytics"
])

with tab1:

    st.subheader("🍽 Find Your Perfect Restaurant")

    st.write("Fill in your preferences and let our AI recommend the best restaurants for you.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        cuisine = st.selectbox(
            "🍜 Preferred Cuisine",
            sorted(dataset["Cuisines"].dropna().unique())
        )

        budget = st.slider(
            "💰 Budget for Two",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )

        table_booking = st.checkbox("🪑 Table Booking Required")

    with col2:

        city = st.selectbox(
            "📍 Select City",
            sorted(dataset["City"].dropna().unique())
        )

        rating = st.slider(
            "⭐ Minimum Rating",
            1.0,
            5.0,
            4.0,
            0.1
        )

        online_delivery = st.checkbox("🚚 Online Delivery")

    st.write("")

    if st.button("🍽 Find Restaurants", use_container_width=True):

        recommendations = recommend_restaurants(
            cuisine=cuisine,
            cost=budget,
            rating=rating,
            city=city,
            table_booking=table_booking,
            online_delivery=online_delivery
        )

        st.success(f"Found {len(recommendations)} Recommended Restaurants")

        st.write("")

        for _, row in recommendations.iterrows():

            with st.container(border=True):

                colA, colB = st.columns([4,1])

                with colA:

                    st.markdown(f"## 🍽 {row['Restaurant Name']}")

                    st.write(f"📍 **City:** {row['City']}")

                    st.write(f"🍜 **Cuisine:** {row['Cuisines']}")

                    st.write(f"⭐ **Rating:** {row['Aggregate rating']}")

                    st.write(f"💰 **Cost for Two:** ₹ {row['Average Cost for two']}")

                    st.write(f"🪑 **Table Booking:** {row['Has Table booking']}")

                    st.write(f"🚚 **Online Delivery:** {row['Has Online delivery']}")

                    st.caption(row["Address"])

                with colB:

                    st.metric(
                        "⭐ Rating",
                        row["Aggregate rating"]
                    )

                st.write("")

with tab2:

    st.subheader("📊 Restaurant Dataset Analytics")

    st.write("Explore insights from the restaurant dataset.")

    st.divider()

    # -------------------------
    # Row 1
    # -------------------------

    col1, col2 = st.columns(2)

    with col1:

        city_counts = (
            dataset["City"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        city_counts.columns = ["City", "Restaurants"]

        fig_city = px.bar(
            city_counts,
            x="City",
            y="Restaurants",
            title="Top 10 Cities by Restaurant Count",
            color="Restaurants",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(
            fig_city,
            use_container_width=True
        )

    with col2:

        rating_fig = px.histogram(
            dataset,
            x="Aggregate rating",
            nbins=20,
            title="Rating Distribution",
            color_discrete_sequence=["#b91c1c"]
        )

        st.plotly_chart(
            rating_fig,
            use_container_width=True
        )

    st.divider()

    # -------------------------
    # Row 2
    # -------------------------

    col3, col4 = st.columns(2)

    with col3:

        cuisine_counts = (
            dataset["Cuisines"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        cuisine_counts.columns = ["Cuisine", "Count"]

        cuisine_fig = px.bar(
            cuisine_counts,
            x="Cuisine",
            y="Count",
            title="Top 10 Cuisine Types",
            color="Count",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(
            cuisine_fig,
            use_container_width=True
        )

    with col4:

        online = dataset["Has Online delivery"].replace(
            {1: "Yes", 0: "No"}
        )

        delivery_fig = px.pie(
            names=online,
            title="Online Delivery Availability",
            hole=0.45
        )

        st.plotly_chart(
            delivery_fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("🏆 Top Rated Restaurants")

    top = dataset.sort_values(
        "Aggregate rating",
        ascending=False
    )[
        [
            "Restaurant Name",
            "City",
            "Cuisines",
            "Aggregate rating",
        ]
    ].head(10)

    st.dataframe(
        top,
        use_container_width=True
    )