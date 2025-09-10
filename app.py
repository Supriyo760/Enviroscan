{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Supriyo760/Enviroscan/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "SQLizy7lhLmE"
      },
      "source": [
        "data set of loader"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "t9TPk-u6c6mX",
        "outputId": "1fe3ed02-66ea-439f-97f8-2682bcea6fac"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Collecting osmnx\n",
            "  Downloading osmnx-2.0.6-py3-none-any.whl.metadata (4.9 kB)\n",
            "Requirement already satisfied: geopandas>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from osmnx) (1.1.1)\n",
            "Requirement already satisfied: networkx>=2.5 in /usr/local/lib/python3.12/dist-packages (from osmnx) (3.5)\n",
            "Requirement already satisfied: numpy>=1.22 in /usr/local/lib/python3.12/dist-packages (from osmnx) (2.0.2)\n",
            "Requirement already satisfied: pandas>=1.4 in /usr/local/lib/python3.12/dist-packages (from osmnx) (2.2.2)\n",
            "Requirement already satisfied: requests>=2.27 in /usr/local/lib/python3.12/dist-packages (from osmnx) (2.32.4)\n",
            "Requirement already satisfied: shapely>=2.0 in /usr/local/lib/python3.12/dist-packages (from osmnx) (2.1.1)\n",
            "Requirement already satisfied: pyogrio>=0.7.2 in /usr/local/lib/python3.12/dist-packages (from geopandas>=1.0.1->osmnx) (0.11.1)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.12/dist-packages (from geopandas>=1.0.1->osmnx) (25.0)\n",
            "Requirement already satisfied: pyproj>=3.5.0 in /usr/local/lib/python3.12/dist-packages (from geopandas>=1.0.1->osmnx) (3.7.2)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas>=1.4->osmnx) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas>=1.4->osmnx) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas>=1.4->osmnx) (2025.2)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests>=2.27->osmnx) (3.4.3)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests>=2.27->osmnx) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests>=2.27->osmnx) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests>=2.27->osmnx) (2025.8.3)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas>=1.4->osmnx) (1.17.0)\n",
            "Downloading osmnx-2.0.6-py3-none-any.whl (101 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m101.5/101.5 kB\u001b[0m \u001b[31m2.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: osmnx\n",
            "Successfully installed osmnx-2.0.6\n"
          ]
        }
      ],
      "source": [
        "!pip install osmnx\n",
        "import pandas as pd\n",
        "import requests\n",
        "import osmnx as ox\n",
        "def fetch_openaq_data(city, params):\n",
        "  url = f\"https://api.openaq.org/v2/measurements?city={city}\"\n",
        "  response = response.get(url, params= params)\n",
        "  data = response.json()['results']\n",
        "  return pd.DataFrame(data)\n",
        "\n",
        "def fetch_weather_data(lat, lon, api_key):\n",
        "  url = f\"https://api.openweathermap.org/data/2.5/weather\"\n",
        "  params = {'lat': lat, 'lon': lon, 'appid': api_key}\n",
        "  response = requests.get(url, params=params)\n",
        "  return response.json()\n",
        "\n",
        "def get_location_features(lat, lon, dist= 1000):\n",
        "  G= ox.graph_from_point((lat, lon), dist= dist, network_type= 'drive')\n",
        "  roads= ox.geometries.geometries_from_point((lat, lon), tags= {'highway': True}, dist=dist)\n",
        "  factories= ox.geometries.geometries_from_point((lat, lon), tags= {'landuse': 'industrial'}, dist= dist)\n",
        "  return {'roads': roads, 'factories': factories}\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8SrpyeXatJzD"
      },
      "source": [
        "data cleaning\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "wTYdWKFYtL30"
      },
      "outputs": [],
      "source": [
        "def clean_pollution_data(df):\n",
        "  df = df.drop_duplicates()\n",
        "  df = df.dropna(subset=['value', 'coordinates.latitude', 'coordinates.longitude'])\n",
        "  df['value']= pd.to_numeric(df['value'])\n",
        "  df['timestamp']= pd.to_datetime(df['date']['utc'])\n",
        "  #impute missing values\n",
        "  df= df.fillna(df.mean())\n",
        "  return df\n",
        "def features_engineering(df):\n",
        "  for col in['value']:\n",
        "    df[col]= (df[col]-df[col].mean())/ df[col].std()\n",
        "    df['hour']= df['timestamp'].dt.hour\n",
        "    df['dayofweek']= df['timestamp'].dt.dayofweek\n",
        "    df['month']= df['timestamp'].dt.month\n",
        "    return df"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "aJJQ_UWLwwn4"
      },
      "source": [
        "SOURCE LABELLING AND SIMULATION\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "VKyF_DdEw0Ja"
      },
      "outputs": [],
      "source": [
        "def label_sources(df):\n",
        "  df['source']= 'Unknown'\n",
        "  df.loc[(df['near_main_road']==1)&(df['NO2']>40), 'source'] = 'Vehicular'\n",
        "  df.loc[(df['near_factory']==1)&(df['SO2']>20), 'source'] = 'Industrial'\n",
        "  df.loc[(df['near_farmland']==1)&(df['season']=='Dry')& (df['PM2.5']>70), 'source'] = 'Agricultural'\n",
        "  return df"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KkS24BDO4QzI"
      },
      "source": [
        "Model Training and Source Prediction"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "CSuYTXwN4WyK"
      },
      "outputs": [],
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "\n",
        "def train_predict_model(df):\n",
        "  features= ['PM2.5','NO2','SO2','CO','roads_proximity','factories_proximity','temperature','humidity','hour','dayofweek','month']\n",
        "  x=df[features]\n",
        "  y=df['source']\n",
        "  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)\n",
        "  cif= RandomForestClassifier()\n",
        "  param_grid= {'n_estimators': [50, 100], 'max_depth': [5,10,None]}\n",
        "  grid= GridSearchCV(cif, param_grid)\n",
        "  grid.fit(x_train, y_train)\n",
        "  y_pred= grid.predict(x_test)\n",
        "  print(classification_report(y_test, y_pred))\n",
        "  return grid.best_estimator_"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "M_hD29AfATEl"
      },
      "source": [
        "Geospatial Mapping and Visualization"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "aFR9rj_5AczR"
      },
      "outputs": [],
      "source": [
        "import folium\n",
        "\n",
        "def plot_heatmap(df):\n",
        "    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)\n",
        "    for _, row in df.iterrows():\n",
        "        folium.Circle(\n",
        "            location=[row['latitude'], row['longitude']],\n",
        "            radius=50,\n",
        "            color=\"red\" if row['source'] == \"Industrial\" else \"blue\",\n",
        "            fill=True\n",
        "        ).add_to(m)\n",
        "    return m\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1pReq5L7AppH"
      },
      "source": [
        "real-time dashboard and alerts(streamlit)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NhPzvr4MAi2d",
        "outputId": "1e475b3e-1148-43b9-b548-f8dd1d4178b9"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.12/dist-packages (1.49.1)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<6,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<7,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.5.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (8.2.1)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.0.2)\n",
            "Requirement already satisfied: packaging<26,>=20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (25.0)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<12,>=7.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (11.3.0)\n",
            "Requirement already satisfied: protobuf<7,>=3.20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.29.5)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.32.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (4.15.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.12/dist-packages (from streamlit) (3.1.45)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.9.1)\n",
            "Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.4.2)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (4.25.1)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (2.3.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.12/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.4.3)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2025.8.3)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (3.0.2)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (25.3.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (2025.4.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (0.36.2)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (0.27.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n",
            "Requirement already satisfied: pyngrok in /usr/local/lib/python3.12/dist-packages (7.3.0)\n",
            "Requirement already satisfied: PyYAML>=5.1 in /usr/local/lib/python3.12/dist-packages (from pyngrok) (6.0.2)\n"
          ]
        }
      ],
      "source": [
        "!pip install streamlit\n",
        "!pip install pyngrok\n",
        "\n",
        "!pip install -q streamlit pyngrok"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KdJ0b5R_Aw7G",
        "outputId": "cc6dc948-dd04-4afd-c863-07e564eb738a"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Overwriting app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "\n",
        "st.set_page_config(page_title=\"Pollution Source Identifier\")\n",
        "\n",
        "st.title(\" AI-Powered Pollution Source Identifier\")\n",
        "\n",
        "city = st.text_input(\"Enter a city name\", placeholder=\"e.g., Delhi\")\n",
        "\n",
        " if st.button(\"Analyze\"):\n",
        "    if city.strip() == \"\":\n",
        "        st.warning(\" Please enter a valid city name.\")\n",
        "    else:\n",
        "        st.success(f\"Analyzing pollution sources for: {city}\")\n",
        "        st.markdown(f\"\"\"\n",
        "        ###  AI Analysis Results for **{city}** (Simulated)\n",
        "        - **Main Pollutants:** PM2.5, NOx, SO2\n",
        "        - **Likely Sources:**\n",
        "            -  Vehicle emissions\n",
        "            - Industrial activity\n",
        "            -  Biomass/garbage burning\n",
        "        - **Air Quality Index (AQI):** 185 (Unhealthy)\n",
        "        - **Recommendation:** Limit outdoor activity. Use masks. Air purifiers recommended indoors.\n",
        "        \"\"\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EqkbQjhhA3xE",
        "outputId": "f0bc1618-9f2a-4478-e3bd-ea8fb809e54c"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "🌐 Your Streamlit app is live at: NgrokTunnel: \"https://7ef3e0b7b2ac.ngrok-free.app\" -> \"http://localhost:8501\"\n"
          ]
        }
      ],
      "source": [
        "# ✅ Step 1: Install required packages\n",
        "!pip install pyngrok streamlit --quiet\n",
        "\n",
        "# ✅ Step 2: Import necessary libraries\n",
        "from pyngrok import ngrok\n",
        "import time\n",
        "import os\n",
        "\n",
        "# ✅ Step 3: Export your ngrok authtoken (this sets an environment variable)\n",
        "os.environ[\"NGROK_AUTHTOKEN\"] =  \"32SlneBxn5rvtKCuoIMTuDOkASV_5YnDn6RerRqkhVmkYQjMv\"\n",
        "\n",
        "# ✅ Step 4: Authenticate pyngrok using the exported token\n",
        "ngrok.set_auth_token(os.environ[\"NGROK_AUTHTOKEN\"])\n",
        "\n",
        "# ✅ Step 5: Kill any existing Streamlit process\n",
        "!pkill streamlit\n",
        "\n",
        "# ✅ Step 6: Define your Streamlit app code\n",
        "app_code = '''\n",
        "import streamlit as st\n",
        "\n",
        "st.set_page_config(page_title=\"Streamlit via ngrok\", page_icon=\"🔗\")\n",
        "st.title(\"🚀 Hello from Streamlit!\")\n",
        "st.write(\"This Streamlit app is running through an ngrok tunnel.\")\n",
        "'''\n",
        "\n",
        "# ✅ Step 7: Write app code to a file\n",
        "with open(\"app.py\", \"w\") as f:\n",
        "    f.write(app_code)\n",
        "\n",
        "# ✅ Step 8: Start the Streamlit app in the background\n",
        "!streamlit run app.py &> /content/logs.txt &\n",
        "\n",
        "# ✅ Step 9: Wait a few seconds for the app to boot\n",
        "time.sleep(5)\n",
        "\n",
        "# ✅ Step 10: Open an ngrok tunnel to port 8501\n",
        "public_url = ngrok.connect(8501)\n",
        "\n",
        "# ✅ Step 11: Print the public URL\n",
        "print(\"🌐 Your Streamlit app is live at:\", public_url)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rY5YCIEhgkpb",
        "outputId": "83ce8f2f-b908-423e-840c-91d2e84d1499"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "🌐 Your Streamlit app is live here: NgrokTunnel: \"https://29e46388b081.ngrok-free.app\" -> \"http://localhost:8501\"\n"
          ]
        }
      ],
      "source": [
        "# 1️⃣ Install required packages\n",
        "!pip install pyngrok streamlit --quiet\n",
        "\n",
        "# 2️⃣ Imports\n",
        "from pyngrok import ngrok\n",
        "import time\n",
        "import os\n",
        "\n",
        "# 3️⃣ Kill any existing Streamlit instance (clean slate)\n",
        "!pkill streamlit\n",
        "\n",
        "# 4️⃣ Set your ngrok authtoken here (replace with your real token)\n",
        "NGROK_AUTHTOKEN = \"32SlneBxn5rvtKCuoIMTuDOkASV_5YnDn6RerRqkhVmkYQjMv\"\n",
        "ngrok.set_auth_token(NGROK_AUTHTOKEN)\n",
        "\n",
        "# 5️⃣ Write your Streamlit app code\n",
        "app_code = '''\n",
        "import streamlit as st\n",
        "\n",
        "st.set_page_config(page_title=\"Streamlit + ngrok\", page_icon=\"🚀\")\n",
        "st.title(\"🚀 Hello from Streamlit!\")\n",
        "st.write(\"This app runs inside Colab and is accessible via ngrok tunnel.\")\n",
        "'''\n",
        "\n",
        "with open(\"app.py\", \"w\") as f:\n",
        "    f.write(app_code)\n",
        "\n",
        "# 6️⃣ Start Streamlit app in background\n",
        "get_ipython().system_raw(\"streamlit run app.py &\")\n",
        "\n",
        "# 7️⃣ Wait for Streamlit server to start\n",
        "time.sleep(5)\n",
        "\n",
        "# 8️⃣ Open ngrok tunnel on port 8501\n",
        "public_url = ngrok.connect(8501)\n",
        "\n",
        "# 9️⃣ Display the public URL\n",
        "print(f\"🌐 Your Streamlit app is live here: {public_url}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "X_YC2C6NsuoL",
        "outputId": "b11b8966-14a1-4a10-ec38-c56b6b06af4d"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "🌐 Your Streamlit app is live here: NgrokTunnel: \"https://377dfc261e85.ngrok-free.app\" -> \"http://localhost:8501\"\n"
          ]
        }
      ],
      "source": [
        "# Start Streamlit in background\n",
        "!streamlit run app.py &> /content/logs.txt &\n",
        "\n",
        "# Give more time for Streamlit to start (e.g., 15 seconds)\n",
        "import time\n",
        "time.sleep(15)\n",
        "\n",
        "# Then create ngrok tunnel\n",
        "from pyngrok import ngrok\n",
        "ngrok.set_auth_token(\"32SlneBxn5rvtKCuoIMTuDOkASV_5YnDn6RerRqkhVmkYQjMv\")\n",
        "public_url = ngrok.connect(8501)\n",
        "print(\"🌐 Your Streamlit app is live here:\", public_url)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "colab": {
          "background_save": True,
          "base_uri": "https://localhost:8080/"
        },
        "id": "K_6yS8apvs1X",
        "outputId": "b324031b-e2eb-40f4-83de-256d93fb8206"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "WARNING:pyngrok.process.ngrok:t=2025-09-09T16:15:57+0000 lvl=warn msg=\"Stopping forwarder\" name=http-8501-0cbe0531-0be3-4d5e-a5d6-ab2abbfc4961 acceptErr=\"failed to accept connection: Listener closed\"\n",
            "WARNING:pyngrok.process.ngrok:t=2025-09-09T16:15:57+0000 lvl=warn msg=\"Stopping forwarder\" name=http-8501-abef5f68-a0a2-4c81-859b-840fd83eda71 acceptErr=\"failed to accept connection: Listener closed\"\n",
            "WARNING:pyngrok.process.ngrok:t=2025-09-09T16:15:57+0000 lvl=warn msg=\"Stopping forwarder\" name=http-8501-26a72d82-acec-4012-b8c6-04de9bbd989b acceptErr=\"failed to accept connection: Listener closed\"\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "🌐 Your Streamlit app is live at: NgrokTunnel: \"https://72ff3d0d8bde.ngrok-free.app\" -> \"http://localhost:8501\"\n"
          ]
        }
      ],
      "source": [
        "# Install required packages (run once)\n",
        "!pip install --quiet streamlit pyngrok\n",
        "\n",
        "import os\n",
        "import time\n",
        "import subprocess\n",
        "from pyngrok import ngrok\n",
        "\n",
        "# Set your ngrok auth token (replace with your actual token)\n",
        "NGROK_AUTH_TOKEN = \"YOUR_NGROK_AUTH_TOKEN_HERE\"\n",
        "ngrok.set_auth_token(NGROK_AUTH_TOKEN)\n",
        "\n",
        "# Write a simple Streamlit app\n",
        "app_code = \"\"\"\n",
        "import streamlit as st\n",
        "\n",
        "st.title(\"🚀 EnviroScan Pollution Source Identifier\")\n",
        "\n",
        "city = st.text_input(\"Enter a city name\", \"Delhi\")\n",
        "if st.button(\"Analyze\"):\n",
        "    if not city.strip():\n",
        "        st.warning(\"Please enter a valid city name.\")\n",
        "    else:\n",
        "        st.success(f\"Analyzing pollution sources for: {city}\")\n",
        "        st.markdown('''\n",
        "        ### AI Analysis Results for **{city}** (Simulated)\n",
        "        - **Main Pollutants:** PM2.5, NOx, SO2\n",
        "        - **Likely Sources:**\n",
        "          - Vehicle emissions\n",
        "          - Industrial activity\n",
        "          - Biomass/garbage burning\n",
        "        - **Air Quality Index (AQI):** 185 (Unhealthy)\n",
        "        - **Recommendation:** Limit outdoor activity. Use masks. Air purifiers recommended indoors.\n",
        "        '''.format(city=city))\n",
        "\"\"\"\n",
        "\n",
        "with open(\"app.py\", \"w\") as f:\n",
        "    f.write(app_code)\n",
        "\n",
        "# Kill previous Streamlit instances\n",
        "subprocess.run([\"pkill\", \"-f\", \"streamlit\"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
        "\n",
        "# Disconnect any existing ngrok tunnels to avoid free-tier limits\n",
        "for tunnel in ngrok.get_tunnels():\n",
        "    ngrok.disconnect(tunnel.public_url)\n",
        "\n",
        "# Start Streamlit app in the background\n",
        "streamlit_process = subprocess.Popen([\"streamlit\", \"run\", \"app.py\"])\n",
        "\n",
        "# Wait for Streamlit server to start\n",
        "time.sleep(15)  # Increase if needed\n",
        "\n",
        "# Open ngrok tunnel to port 8501\n",
        "public_url = ngrok.connect(8501)\n",
        "print(f\"🌐 Your Streamlit app is live at: {public_url}\")\n",
        "\n",
        "# Keep process alive to maintain server & tunnel\n",
        "try:\n",
        "    streamlit_process.wait()\n",
        "except KeyboardInterrupt:\n",
        "    streamlit_process.terminate()\n",
        "    ngrok.disconnect(public_url)\n",
        "    print(\"Terminated Streamlit and ngrok tunnel.\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {
        "id": "mNMdYEphxuK8"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO0zD2Y/Kyo8ldO/BYjf9jG",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
