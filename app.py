from src.ml_train import train_model



import os, joblib, pandas as pd, streamlit as st
import plotly.graph_objects as go
from datetime import date
from src.config import MODEL_PATH, LATE_THRESHOLD_MIN
from src.utils import fetch_query
from PIL import Image


# Set page config with favicon
st.set_page_config(
    page_title='Blinkit Decision Platform',
    layout='wide',
    page_icon='favicon-96x96.png'  # Use the local favicon
)

# Add background color and style for the main area and title, with an icon
st.markdown('''
    <style>
    .main-bg {
        background: linear-gradient(120deg, #f3f7fa 0%, #e8f5e9 100%) !important;
        padding: 32px 32px 8px 32px;
        border-radius: 18px;
        margin-bottom: 18px;
            box-shadow: 0 4px 24px rgba(0,158,96,0.08), 0 1.5px 0 #b2dfdb;
            border: 2px solid #b2dfdb;
    }
    .main-title-row {
        display: flex;
        align-items: center;
        gap: 18px;
            margin-bottom: 4px;
    }
    .main-title-icon {
        font-size: 3.0rem;
        margin-right: 6px;
            filter: drop-shadow(0 2px 4px #b2dfdb);
    }
    .main-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #009e60;
        letter-spacing: 0.5px;
            text-shadow: 0 1px 0 #fff, 0 2px 8px #b2dfdb44;
    }
        .main-subtitle {
            font-size: 1.05rem;
            color: #555;
            font-weight: 400;
            margin-left: 2.7rem;
            margin-top: 2px;
            margin-bottom: 0;
            letter-spacing: 0.1px;
        }
    </style>
    <div class="main-bg">
        <div class="main-title-row">
            <span class="main-title-icon">💡</span>
            <span class="main-title">AI-Powered Blinkit Business Decision Platform</span>
        </div>
            <div class="main-subtitle">Empowering data-driven decisions for modern retail operations</div>
    </div>
''', unsafe_allow_html=True)


st.sidebar.markdown(
    """
    
    <div style='text-align:center; margin-bottom: 0.3rem;'></div>
        <style>
    .sidebar-nav-title {
        display: flex;
        align-items: center;
        font-size: 1.18rem;
        font-weight: 700;
        color: #009e60;
        margin-bottom: 1.5rem;
        gap: 8px;
    }
    </style>
    <div class="sidebar-nav-title">🧭 Navigation</div>
    """,
    unsafe_allow_html=True
    )
page = st.sidebar.radio('Go to', [
    '📊 Dashboard',
    '🛠️ Model Training',
    '🛵 Delay Risk Calculator',
    '🤖 AI Business Assistant (RAG)'
])

# Sidebar date filters (only for Dashboard)
if page == '📊 Dashboard':
    st.sidebar.header('Filters')
    start_date = st.sidebar.date_input('Start date', value=date(2024, 11, 1))
    end_date   = st.sidebar.date_input('End date', value=date(2024, 12, 31))



    

# Dashboard Page
if page == '📊 Dashboard':
    # Fetch master data from DB
    @st.cache_data
    def load_master_from_db(start_date, end_date):
        query = '''
            SELECT
                o.order_date AS day,
                COUNT(o.order_id) AS orders_count,
                SUM(o.order_total) AS total_revenue,
                COALESCE(SUM(m.spend), 0) AS total_spend,
                COALESCE(SUM(m.impressions), 0) AS total_impressions,
                GROUP_CONCAT(DISTINCT m.channel) AS channels
            FROM blinkit_orders o
            LEFT JOIN blinkit_marketing_performance m ON DATE(o.order_date) = m.date
            WHERE o.order_date BETWEEN %s AND %s
            GROUP BY day
            ORDER BY day
        '''
        df = fetch_query(query, params=(start_date, end_date))
        df['day'] = pd.to_datetime(df['day'])
        df['roas'] = df.apply(lambda r: r['total_revenue']/r['total_spend'] if r['total_spend'] > 0 else None, axis=1)
        return df

    master = load_master_from_db(start_date, end_date)
    master_f = master

    # Enhanced KPI cards with gradient, icon, and hover effect
    st.markdown("""
        <style>
        .kpi-card {
            background: linear-gradient(120deg, #e8f5e9 0%, #f3f7fa 100%);
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.09);
            padding: 28px 0 18px 0;
            margin-bottom: 22px;
            text-align: center;
            transition: box-shadow 0.2s, transform 0.2s;
            border: 1.5px solid #e0e0e0;
        }
        .kpi-card:hover {
            box-shadow: 0 8px 32px rgba(0,200,83,0.13);
            transform: translateY(-2px) scale(1.03);
            border: 1.5px solid #b2dfdb;
        }
        .kpi-label {
            color: #009e60;
            font-size: 1.13rem;
            font-weight: 700;
            margin-bottom: 7px;
            letter-spacing: 0.2px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
        }
        .kpi-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #222;
            margin-bottom: 2px;
        }
        .kpi-icon {
            font-size: 1.3em;
            vertical-align: middle;
        }
        </style>
    """, unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label"><span class="kpi-icon">💰</span> Total Revenue <span title="Sum of all order revenue">ℹ️</span></div><div class="kpi-value">₹{master_f["total_revenue"].sum():,.0f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label"><span class="kpi-icon">📢</span> Total Spend <span title="Sum of all marketing spend">ℹ️</span></div><div class="kpi-value">₹{master_f["total_spend"].sum():,.0f}</div></div>', unsafe_allow_html=True)
    with c3:
        roas = f'{master_f["roas"].mean():.2f}' if 'roas' in master_f and not master_f['roas'].isna().all() else 'N/A'
        st.markdown(f'<div class="kpi-card"><div class="kpi-label"><span class="kpi-icon">📈</span> Avg ROAS <span title="Return on Ad Spend">ℹ️</span></div><div class="kpi-value">{roas}</div></div>', unsafe_allow_html=True)
    with c4:
        orders = f'{int(master_f["orders_count"].sum())}' if 'orders_count' in master_f else 'N/A'
        st.markdown(f'<div class="kpi-card"><div class="kpi-label"><span class="kpi-icon">🛒</span> Orders <span title="Total number of orders">ℹ️</span></div><div class="kpi-value">{orders}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr style="margin: 30px 0 20px 0; border: none; border-top: 2px solid #e0e0e0;">', unsafe_allow_html=True)

    # Dual-axis: Revenue vs Spend
    with st.expander('📊 Marketing ROI (ROAS): Revenue vs Spend', expanded=True):
        fig = go.Figure()
        if not master_f.empty:
            fig.add_trace(go.Scatter(x=master_f['day'], y=master_f['total_revenue'], name='Revenue',
                                     mode='lines+markers', line=dict(color='green')))
            fig.add_trace(go.Bar(x=master_f['day'], y=master_f['total_spend'], name='Ad Spend',
                                 marker_color='red', yaxis='y2'))
            fig.update_layout(xaxis_title='Date',
                              yaxis=dict(title='Revenue', showgrid=True),
                              yaxis2=dict(title='Ad Spend', overlaying='y', side='right'),
                              legend=dict(orientation='h'), height=450,
                              plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No data to display.')



    

# Model Training Page
elif page == '🛠️ Model Training':
    st.markdown('<h2 style="color:#009e60;display:flex;align-items:center;gap:10px;">🛠️ Model Training <span style="font-size:1.1rem; color:#888;">(Delay Risk)</span></h2>', unsafe_allow_html=True)
    st.info('Click the button below to retrain the delay risk model using the latest data.', icon="ℹ️")
    if st.button('🔄 Retrain Delay Risk Model'):
        with st.spinner('Training model...'):
            try:
                auc = train_model()
                st.success(f'Model retrained successfully! AUC = {auc:.3f}')
                st.cache_resource.clear()  # Clear model cache to reload
            except Exception as e:
                st.error(f'Error during training: {e}')


    

# Delay Risk Calculator Page
elif page == '🛵 Delay Risk Calculator':
    st.markdown('<h2 style="color:#009e60;display:flex;align-items:center;gap:10px;">🛵 Delay Risk Calculator</h2>', unsafe_allow_html=True)
    st.caption('Estimate the risk of delivery delay for a given region, day, and hour.')

    @st.cache_resource
    def load_model(model_path):
        return joblib.load(model_path) if os.path.exists(model_path) else None

    bundle = load_model(MODEL_PATH)
    if bundle is None:
        st.warning('Model not found. Train it via: python src/ml_train.py')
    else:
        clf = bundle['pipeline']; auc = bundle.get('auc')
        if auc: st.caption(f"Model AUC: {auc:.3f}")

        with st.form('risk_form'):
            region = st.selectbox('Region', options=['Indiranagar', 'Koramangala', 'Whitefield', 'Unknown'], help="Select the delivery region.")
            day_of_week = st.selectbox('Day of Week (Mon=0)', options=list(range(7)), help="0=Monday, 6=Sunday")
            hour_of_day = st.slider('Hour of Day', min_value=0, max_value=23, value=18, help="Select the hour of the day.")
            submitted = st.form_submit_button('Calculate Risk')
        if submitted:
            X = pd.DataFrame({'region':[region], 'day_of_week':[day_of_week], 'hour_of_day':[hour_of_day]})
            prob = clf.predict_proba(X)[:, 1][0]
            risk = int(round(prob*100))
            if risk < 40:
                st.success(f'✅ Low Risk of Delay ({risk}%)')
            elif risk < 70:
                st.warning(f'⚠️ Medium Risk of Delay ({risk}%)')
            else:
                st.error(f'⚠️ High Risk of Delay ({risk}%)')


    

# AI Business Assistant (RAG) Page
elif page == '🤖 AI Business Assistant (RAG)':
    st.markdown('<h2 style="color:#009e60;display:flex;align-items:center;gap:10px;">🤖 AI Business Assistant <span style="font-size:1.1rem; color:#888;">(RAG)</span></h2>', unsafe_allow_html=True)
    st.caption('Ask: “Why did sales drop yesterday?” or “Why are customers angry?”')

    with st.expander('💬 Ask a business question (RAG)', expanded=True):
        # Hugging Face Transformers + SentenceTransformers RAG (no API key needed)
        try:
            from transformers import pipeline
            from sentence_transformers import SentenceTransformer, util

            @st.cache_resource
            def load_local_llm():
                # Use a small, CPU-friendly model for local inference
                llm = pipeline("text-generation", model="distilgpt2")
                embedder = SentenceTransformer('all-MiniLM-L6-v2')
                return llm, embedder

            llm, embedder = load_local_llm()

            feedback_df = fetch_query("SELECT feedback_text FROM blinkit_customer_feedback WHERE feedback_text IS NOT NULL")
            texts = feedback_df['feedback_text'].dropna().tolist() if not feedback_df.empty else []
            if not texts:
                st.info('No feedback data found in blinkit_customer_feedback table.')
            else:
                q = st.text_input('Your question', help="Ask about sales, customer feedback, etc.")
                if q:
                    question_emb = embedder.encode(q, convert_to_tensor=True)
                    feedback_embs = embedder.encode(texts, convert_to_tensor=True)
                    hits = util.semantic_search(question_emb, feedback_embs, top_k=4)[0]
                    context = "\n\n".join([texts[hit['corpus_id']] for hit in hits])
                    st.write('**Relevant feedback:**', *[f"- {texts[hit['corpus_id']][:250]}" for hit in hits])

                    prompt = f"Question: {q}\nContext:\n{context}\nAnswer:"
                    result = llm(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
                    st.markdown('**Assistant Answer:**')
                    st.write(result[0]['generated_text'].split('Answer:')[-1].strip())
        except Exception as e:
            st.warning(f'RAG features are optional and need Hugging Face Transformers/SentenceTransformers. ({e})')

# Footer with credits/branding
st.markdown("""
    <hr style='margin-top:40px; margin-bottom:10px; border: none; border-top: 1.5px solid #b2dfdb;'>
    <div style='text-align:center; color:#888; font-size:1rem;'>
        Made with <span style='color:#009e60;'>Streamlit</span> | <a href='https://blinkit.com' style='color:#009e60;text-decoration:none;' target='_blank'>Blinkit IQ</a> &copy; 2025
    </div>
    """, unsafe_allow_html=True)
