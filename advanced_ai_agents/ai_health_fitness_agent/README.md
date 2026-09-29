# 🏋️ AI Health & Fitness Planner Agent

A multi-agent AI application that generates general nutrition and fitness plans based on user-provided information.

The application uses two specialized AI agents — a **Diet Agent** and a **Fitness Agent** — powered by **Agno and Groq**, with a simple **Streamlit** interface.

> Part of **Unwinding-LLM → Advanced AI Agents**

---

## ✨ Features

- 🥗 Personalized general nutrition suggestions
- 🏋️ Personalized general workout suggestions
- 👤 Uses age, height, weight, and activity level
- 🎯 Supports different fitness goals
- 🥑 Supports different dietary preferences
- 💪 Supports different workout preferences
- 🤖 Two specialized AI agents
- 🌐 Simple Streamlit interface
- ⚡ Groq-powered LLM inference
- 🔐 API key stored using environment variables

---

## 🏗️ Architecture

```text
                         User
                           │
                           ▼
                    Streamlit UI
                           │
                    User Information
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       🥗 Diet Agent               🏋️ Fitness Agent
             │                           │
             └─────────────┬─────────────┘
                           ▼
                     Groq LLM
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Nutrition Plan              Fitness Plan
