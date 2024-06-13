import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)

buzzer = 17
GPIO.setup(buzzer, GPIO.OUT)
GPIO.output(buzzer, GPIO.HIGH)  # Stop the beep

def beep(duration):
    GPIO.output(buzzer, GPIO.LOW)  #
    sleep(duration)  
    GPIO.output(buzzer, GPIO.HIGH)
    
def beep_sos():
    # 3 short beeps
    for _ in range(3):
        beep(0.2)
        sleep(0.2)

    # Pause between the letters S and O
    sleep(0.5)

    # 3 long beeps
    for _ in range(3):
        beep(0.5)
        sleep(0.2)

    # Pause between the letters O and S
    sleep(0.5)

    # 3 short beeps again
    for _ in range(3):
        beep(0.2)
        sleep(0.2)

    # Pause before the signal repeats
    sleep(2)
    