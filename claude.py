"""
TMC2209 Stepper Motor Control — Le Potato (AML-S905X-CC)
---------------------------------------------------------
All 40-pin header GPIO lines live on gpiochip1 (periphs-banks).
Pin numbers below are gpiochip1 LINE OFFSETS, not BCM or physical pin numbers.

Wiring (all on gpiochip1):
  Motor 1: EN -> Pin7  (line 98) | STEP -> Pin8  (line 91) | DIR -> Pin10 (line 92)
  Motor 2: EN -> Pin16 (line 93) | STEP -> Pin18 (line 94) | DIR -> Pin22 (line 79)
  Motor 3: EN -> Pin29 (line 96) | STEP -> Pin31 (line 97) | DIR -> Pin32 (line 95)
  Motor 4: EN -> Pin33 (line 85) | STEP -> Pin35 (line 86) | DIR -> Pin36 (line 81)

TMC2209 notes:
  - EN is active LOW  (0 = motor enabled, 1 = motor disabled/freewheel)
  - DIR setup time    >= 20us before first STEP pulse
  - STEP pulse width  >= 100ns (Python toggle is always safely above this)
  - Microsteps set by MS1/MS2 pins on the breakout board (floating = 1/8 typical)

Requirements:
  pip install gpiod
"""

import gpiod
import time

# -- GPIO Chip ----------------------------------------------------------------
GPIO_CHIP = "gpiochip1"

# -- Motor Pin Definitions (gpiochip1 line offsets) ---------------------------
# Change these if you wire differently — use the line numbers from gpioinfo.
MOTORS = {
    "motor1": {"en": 98, "step": 91, "dir": 92},
    "motor2": {"en": 93, "step": 94, "dir": 79},
    "motor3": {"en": 96, "step": 97, "dir": 95},
    "motor4": {"en": 85, "step": 86, "dir": 81},
}

# -- Motion Parameters --------------------------------------------------------
MICROSTEPS     = 8    # Match your MS1/MS2 pin strapping on the TMC2209 board
                      # Common: floating=8, MS1=16, MS2=32, MS1+MS2=64
STEPS_PER_REV  = 200  # Standard 1.8 degree stepper = 200 full steps/rev
USTEPS_PER_REV = STEPS_PER_REV * MICROSTEPS

DEFAULT_RPM    = 60
MIN_STEP_DELAY = 0.0001  # 100us floor — below this Python timing gets unreliable


def rpm_to_step_delay(rpm: float) -> float:
    """Convert RPM to the inter-step delay in seconds."""
    usteps_per_sec = (rpm / 60.0) * STEPS_PER_REV * MICROSTEPS
    return 1.0 / usteps_per_sec


class TMC2209Motor:
    """One stepper motor driven by a TMC2209 over STEP/DIR/EN."""

    def __init__(self, chip: gpiod.Chip, name: str, pins: dict):
        self.name = name

        self.en_line   = chip.get_line(pins["en"])
        self.step_line = chip.get_line(pins["step"])
        self.dir_line  = chip.get_line(pins["dir"])

        consumer = f"tmc2209-{name}"
        # EN starts HIGH = disabled (TMC2209 active-low enable)
        self.en_line.request(  consumer=consumer, type=gpiod.LINE_REQ_DIR_OUT, default_val=1)
        self.step_line.request(consumer=consumer, type=gpiod.LINE_REQ_DIR_OUT, default_val=0)
        self.dir_line.request( consumer=consumer, type=gpiod.LINE_REQ_DIR_OUT, default_val=0)

        print(f"[{self.name}] Initialized (disabled).")

    # -- Enable / Disable -----------------------------------------------------

    def enable(self):
        """Pull EN LOW -> driver active, holding torque on."""
        self.en_line.set_value(0)
        time.sleep(0.001)  # 1ms settle after enable

    def disable(self):
        """Pull EN HIGH -> driver off, motor freewheels, driver stays cool."""
        self.en_line.set_value(1)

    # -- Direction ------------------------------------------------------------

    def set_direction(self, clockwise: bool = True):
        """
        Set rotation direction.
        If your motor runs the wrong way, swap True/False at the call site,
        or swap any two motor coil wires on the connector.
        """
        self.dir_line.set_value(0 if clockwise else 1)
        time.sleep(0.00002)  # TMC2209 requires >= 20us DIR setup time

    # -- Stepping -------------------------------------------------------------

    def _step(self, delay: float):
        """Send one STEP pulse then wait `delay` seconds."""
        self.step_line.set_value(1)
        self.step_line.set_value(0)
        time.sleep(delay)

    # -- Public Move API ------------------------------------------------------

    def move_steps(self, steps: int, rpm: float = DEFAULT_RPM, clockwise: bool = True):
        """Move a raw number of microsteps."""
        delay = max(rpm_to_step_delay(rpm), MIN_STEP_DELAY)
        self.set_direction(clockwise)
        self.enable()
        print(f"[{self.name}] {steps} microsteps @ {rpm} RPM ({'CW' if clockwise else 'CCW'})")
        for _ in range(steps):
            self._step(delay)

    def move_degrees(self, degrees: float, rpm: float = DEFAULT_RPM, clockwise: bool = True):
        """Move by a number of degrees."""
        steps = round((degrees / 360.0) * USTEPS_PER_REV)
        self.move_steps(steps, rpm, clockwise)

    def move_revolutions(self, revs: float, rpm: float = DEFAULT_RPM, clockwise: bool = True):
        """Move by a number of full revolutions."""
        steps = round(revs * USTEPS_PER_REV)
        self.move_steps(steps, rpm, clockwise)

    # -- Cleanup --------------------------------------------------------------

    def release(self):
        """Disable and release all GPIO lines."""
        self.disable()
        self.en_line.release()
        self.step_line.release()
        self.dir_line.release()
        print(f"[{self.name}] Released.")


# -- Main ---------------------------------------------------------------------

def main():
    chip = gpiod.Chip(GPIO_CHIP)

    motors = {
        name: TMC2209Motor(chip, name, pins)
        for name, pins in MOTORS.items()
    }

    m1 = motors["motor1"]
    m2 = motors["motor2"]
    m3 = motors["motor3"]
    m4 = motors["motor4"]

    i = 0
    while i < 10:
        i+=1


        m1.move_revolutions(10.0, rpm=60, clockwise=True)
        m1.disable()
        time.sleep(1)
    






    for motor in motors.values():
            motor.release()
            chip.close()


if __name__ == "__main__":
    main()
       