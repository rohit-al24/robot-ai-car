#!/usr/bin/env python3
"""
Motor Driver Controller for 4WD Autonomous Robot Car.
Supports L298N, L293D, TB6612FNG, or dual H-bridges with PWM speed control.
Hardware-agnostic: falls back to simulated GPIO when testing on PC/Mac.
"""
import time

# Default Raspberry Pi GPIO Pinout (BCM Numbering)
# Left Motors (Front-Left + Rear-Left wired in parallel)
PIN_ENA = 12   # Hardware PWM0 (GPIO 12)
PIN_IN1 = 17   # Direction 1
PIN_IN2 = 27   # Direction 2

# Right Motors (Front-Right + Rear-Right wired in parallel)
PIN_ENB = 13   # Hardware PWM1 (GPIO 13)
PIN_IN3 = 22   # Direction 3
PIN_IN4 = 23   # Direction 4

# Optional: Set to False if motors turn backwards
LEFT_INVERT = False
RIGHT_INVERT = False


class MotorDriver:
    def __init__(self, ena=PIN_ENA, in1=PIN_IN1, in2=PIN_IN2,
                 enb=PIN_ENB, in3=PIN_IN3, in4=PIN_IN4):
        self.ena_pin = ena
        self.in1_pin = in1
        self.in2_pin = in2
        self.enb_pin = enb
        self.in3_pin = in3
        self.in4_pin = in4

        self.pwm_a = None
        self.pwm_b = None
        self.is_real_pi = False

        self._init_gpio()

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(GPIO.BCM)
            self.GPIO.setwarnings(False)

            # Setup output pins
            for pin in (self.ena_pin, self.in1_pin, self.in2_pin,
                        self.enb_pin, self.in3_pin, self.in4_pin):
                self.GPIO.setup(pin, GPIO.OUT)
                self.GPIO.output(pin, GPIO.LOW)

            # Setup 1kHz PWM for smooth motor speed control
            self.pwm_a = self.GPIO.PWM(self.ena_pin, 1000)
            self.pwm_b = self.GPIO.PWM(self.enb_pin, 1000)
            self.pwm_a.start(0)
            self.pwm_b.start(0)
            self.is_real_pi = True
            print("[MotorDriver] RPi.GPIO initialized successfully with hardware PWM.")
        except Exception as e:
            print(f"[MotorDriver Note] Running in simulation mode ({e})")
            self.is_real_pi = False

    def _set_motor(self, in1, in2, pwm, speed):
        """
        speed: -100 to +100
        """
        duty = min(100, max(0, abs(speed)))
        if self.is_real_pi:
            pwm.ChangeDutyCycle(duty)
            if speed > 5:
                self.GPIO.output(in1, self.GPIO.HIGH)
                self.GPIO.output(in2, self.GPIO.LOW)
            elif speed < -5:
                self.GPIO.output(in1, self.GPIO.LOW)
                self.GPIO.output(in2, self.GPIO.HIGH)
            else:
                self.GPIO.output(in1, self.GPIO.LOW)
                self.GPIO.output(in2, self.GPIO.LOW)

    def drive(self, left_speed, right_speed):
        """
        Controls left and right motor banks.
        left_speed, right_speed: -100 (full reverse) to +100 (full forward)
        """
        if LEFT_INVERT:
            left_speed = -left_speed
        if RIGHT_INVERT:
            right_speed = -right_speed

        if self.is_real_pi:
            self._set_motor(self.in1_pin, self.in2_pin, self.pwm_a, left_speed)
            self._set_motor(self.in3_pin, self.in4_pin, self.pwm_b, right_speed)

    def forward(self, speed=60):
        self.drive(speed, speed)

    def backward(self, speed=60):
        self.drive(-speed, -speed)

    def steer(self, throttle=50, steering=0.0):
        """
        Differential steering control.
        throttle: forward speed (0 to 100)
        steering: -1.0 (hard left) to +1.0 (hard right), 0.0 = straight
        """
        # Differential calculation
        left = throttle * (1.0 + steering)
        right = throttle * (1.0 - steering)

        # Normalize to max 100
        max_val = max(abs(left), abs(right), 100.0)
        if max_val > 100.0:
            left = (left / max_val) * 100.0
            right = (right / max_val) * 100.0

        self.drive(left, right)

    def spin_left(self, speed=50):
        self.drive(-speed, speed)

    def spin_right(self, speed=50):
        self.drive(speed, -speed)

    def stop(self):
        self.drive(0, 0)

    def cleanup(self):
        self.stop()
        if self.is_real_pi:
            if self.pwm_a:
                self.pwm_a.stop()
            if self.pwm_b:
                self.pwm_b.stop()
            self.GPIO.cleanup()
            print("[MotorDriver] GPIO Cleaned up.")


if __name__ == "__main__":
    motors = MotorDriver()
    print("Testing Motors: Forward 2s -> Left 1s -> Right 1s -> Stop")
    try:
        motors.forward(50)
        time.sleep(2)
        motors.spin_left(50)
        time.sleep(1)
        motors.spin_right(50)
        time.sleep(1)
        motors.stop()
    finally:
        motors.cleanup()
