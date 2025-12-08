import RPi.GPIO as GPIO
import time

# Use Broadcom pin-numbering
GPIO.setmode(GPIO.BCM)

BUZZER_PIN = 2
GPIO.setup(BUZZER_PIN, GPIO.OUT)

def beep(duration=3):
    """
    Turn the buzzer on for `duration` seconds, then off.
    duration: float or int, seconds to buzz (e.g. 3–5)
    """
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

try:
    print("SEVERE DROWSINESS DETECTED - CONTINUOUS ALARM ACTIVATED")
    # Buzz for 5 seconds (or use any value between 3 and 5)
    beep(3)
finally:
    # Always clean up GPIO state on exit
    GPIO.cleanup()
