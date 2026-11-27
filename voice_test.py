import pyttsx3

print("CONTEXTRA VOICE TEST STARTED")

try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.say("Wake up, wake up! Please stay alert.")
    engine.runAndWait()
    engine.stop()
    print("Voice test finished.")
except Exception as e:
    print("VOICE ERROR:", e)

input("Press Enter to close this test...")
