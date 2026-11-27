"""
Vehicle response logic for the SIMULATED ego car (never a real vehicle).

    NORMAL        drive at the scenario speed
    SLOWING       risk is ELEVATED/HIGH: reduce speed
    PULLING_OVER  HIGH risk persisted: decelerate and move to the shoulder
    STOPPED       stopped on the shoulder; resumes only after the driver has
                  been clearly alert for a while

Pure Python so it can be unit-tested.
"""

NORMAL, SLOWING, PULLING_OVER, STOPPED = "NORMAL", "SLOWING", "PULLING OVER", "STOPPED"


class Responder:
    def __init__(self, high_to_pullover_s=3.0, alert_resume_s=8.0, slow_factor=0.7):
        self.high_to_pullover_s = high_to_pullover_s
        self.alert_resume_s = alert_resume_s
        self.slow_factor = slow_factor
        self.mode = NORMAL
        self._high_timer = 0.0
        self._alert_timer = 0.0

    @property
    def speed_factor(self):
        return {NORMAL: 1.0, SLOWING: self.slow_factor, PULLING_OVER: 0.0, STOPPED: 0.0}[self.mode]

    @property
    def wants_shoulder(self):
        return self.mode in (PULLING_OVER, STOPPED)

    def update(self, dt, level, driver_alert_ok, speed_kmh):
        """level: LOW/ELEVATED/HIGH or None (no driver evidence). Returns the mode."""
        if self.mode in (NORMAL, SLOWING):
            if level == "HIGH":
                self._high_timer += dt
            else:
                self._high_timer = max(0.0, self._high_timer - dt)

            if self._high_timer >= self.high_to_pullover_s:
                self.mode = PULLING_OVER
                self._alert_timer = 0.0
            elif level in ("HIGH", "ELEVATED"):
                self.mode = SLOWING
            else:
                self.mode = NORMAL
            return self.mode

        # PULLING_OVER or STOPPED
        if self.mode == PULLING_OVER and speed_kmh <= 1.0:
            self.mode = STOPPED

        self._alert_timer = self._alert_timer + dt if driver_alert_ok else 0.0
        if self._alert_timer >= self.alert_resume_s:
            self.mode = NORMAL
            self._high_timer = 0.0
            self._alert_timer = 0.0
        return self.mode

    @property
    def resume_progress(self):
        """0..1 progress towards resuming (shown on screen while stopped)."""
        return min(1.0, self._alert_timer / self.alert_resume_s)
