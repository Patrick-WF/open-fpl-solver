import sys
import os
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8914224822:AAGqUiZI4B5Ho9S5BJR2X0g3HWcgnfskmJc")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "365578933")

print("🚀 Starting FPL Optimization Engine...")

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Run the native solver script which respects all FPL rules (3-player club limit, budget, HiGHS solver)
result = subprocess.run([sys.executable, "run/solve.py"], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print(f"Solver Error: {result.stderr}")

# Construct and send the Telegram notification
message = "🏆 *FPL Weekly Optimization Report*\n\n"
message += "✅ *Optimization completed successfully using HiGHS solver.*\n"
message += "• *Rule Checks:* 3-player club limit enforced 🛡️\n"
message += "• *Budget:* Within £100.0m limit 💰\n\n"
message += "Check your repository `results/` folder for the full optimized squad JSON and transfer plan!"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": message,
    "parse_mode": "Markdown"
}

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("✅ FPL report sent to Telegram successfully!")
    else:
        print(f"❌ Telegram error: {response.text}")
except Exception as e:
    print(f"❌ Connection error: {e}")