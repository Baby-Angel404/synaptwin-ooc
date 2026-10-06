import urllib.parse
import urllib.request

url = "https://docs.google.com/forms/d/e/1FAIpQLSdRAat5jIunRaFNh_NntsVeJUnekEJDrbuokLZ32LFgCwPtiA/formResponse"

payload = {
    "entry.542177303": "SynapTwin AI BioLab",
    "entry.574430931": "Simon Marc",
    "entry.2024847943": "Individual",
    "entry.233117925": "Aegis / Independent BioAI Lab",
    "entry.1030185898": "dev@aegis-browser.org",
    "entry.217626683": "+6281234567890",
    "entry.882706095": "Indonesia",
    "entry.974061862": "Simon Marc",
    "entry.260764289": "Not sure yet",
    "entry.441796381": "simonmarc",
}

data = urllib.parse.urlencode(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})

try:
    with urllib.request.urlopen(req) as resp:
        print("HTTP Status:", resp.status)
        if resp.status == 200:
            print("[✓] Google Form registration successfully submitted!")
except Exception as e:
    print("Submission Error:", e)
