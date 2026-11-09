# Streamlit Cloud Deployment Guide

## ✅ Deploy to Streamlit Cloud in 3 Steps

### Step 1 — Push to GitHub

```bash
cd "Quadrotor Intelligent Control System"
git init
git add .
git commit -m "Initial commit: Quadrotor ISC Streamlit Cloud App"
git remote add origin https://github.com/YOUR_USERNAME/quadrotor-isc.git
git push -u origin main
```

### Step 2 — Connect on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Select your GitHub repository
4. Set **Main file path** → `streamlit_app.py`
5. Click **Deploy**

### Step 3 — Done!

Your app will be live at:
```
https://YOUR_USERNAME-quadrotor-isc-streamlit-app-XXXXX.streamlit.app
```

---

## 📁 Cloud Deployment Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | ✅ Main app — self-contained, no pygame |
| `requirements.txt` | ✅ Cloud-safe dependencies (no pygame) |
| `.streamlit/config.toml` | ✅ Dark aerospace theme |

---

## 🏃 Run Locally

```bash
# Local Streamlit
streamlit run streamlit_app.py

# Or via main.py
python main.py --dashboard
```

---

## ⚠️ Notes

- `pygame` is **excluded** from `requirements.txt` — it is only needed for the local `python main.py` Pygame simulation window, which cannot run on cloud servers.
- All physics, PID control, and visualization in `streamlit_app.py` are **fully self-contained** (no imports from local modules).

