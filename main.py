# app.py
import os
import requests
import streamlit as st

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

# --------------------------------------------------
# LOAD ENV VARIABLES
# --------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not GROQ_API_KEY or not OPENWEATHER_API_KEY:
    st.error("❌ Missing API keys. Check your .env file.")
    st.stop()

# --------------------------------------------------
# STREAMLIT PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="🌦️ Weather Assistant",
    page_icon="🌤️",
    layout="centered",
)

# Custom CSS
st.markdown("""
<style>
    .stTextInput > label { font-size: 20px; }
    .stMarkdown { font-size: 30px; }
    h1 { font-size: 48px; }
    .stButton > button { font-size: 18px; height: 60px; }
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather Assistant")
#st.caption("Powered by Pydantic-AI, Groq LLaMA 3, and OpenWeatherMap")

# --------------------------------------------------
# Pydantic Output Schema
# --------------------------------------------------
class WeatherForecast(BaseModel):
    location: str
    description: str
    temperature_celsius: float
    humidity: int
    wind_speed: float
    feels_like: float

# --------------------------------------------------
# AI AGENT SETUP
# --------------------------------------------------
weather_agent = Agent(
    model="groq:llama-3.3-70b-versatile",
    model_settings=ModelSettings(temperature=0.2),
    output_type=str,
    system_prompt=(
        "You are a friendly weather assistant. "
        "Always use the get_weather_forecast tool to get real-time weather data. "
        "After presenting the weather, give 2–3 practical recommendations "
        "(e.g., umbrella, hydration, clothing advice). "
        "Never invent weather data."
    ),
)

# --------------------------------------------------
# TOOL: OpenWeatherMap API
# --------------------------------------------------
@weather_agent.tool
def get_weather_forecast(ctx: RunContext, city: str) -> WeatherForecast:
    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city.strip(),
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if response.status_code != 200:
        raise ValueError(data.get("message", "Unable to fetch weather data"))

    return WeatherForecast(
        location=f"{data['name']}, {data['sys']['country']}",
        description=data["weather"][0]["description"].capitalize(),
        temperature_celsius=round(data["main"]["temp"], 1),
        humidity=data["main"]["humidity"],
        wind_speed=round(data["wind"]["speed"], 1),
        feels_like=round(data["main"]["feels_like"], 1),
    )

# --------------------------------------------------
# UI INPUT
# --------------------------------------------------
user_question = st.text_input(
    "Ask about the weather 🌍",
    placeholder="What's the city?",
)

# --------------------------------------------------
# RUN AGENT
# --------------------------------------------------
if st.button("Get Weather 🌤️"):
    if not user_question.strip():
        st.warning("⚠️ Please enter a city or weather question.")
    else:
        with st.spinner("🔍 Fetching weather information..."):
            try:
                result = weather_agent.run_sync(user_question)
                st.success("✅ Current Weather")
                st.markdown(result.output)
            except Exception as e:
                st.error(f"❌ {str(e)}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
#st.caption("Built with ❤️ using Streamlit & Pydantic-AI")






