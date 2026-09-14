# server.py
import math

import simpy


class Server:
    def __init__(
        self,
        env,
        server_type,
        server_id,
        processing_frequency,
        base_failure_rate,
        latitude,
        longitude,
        uplink_rate_mbps=20.0,
    ):
        self.env = env
        self.server_type = server_type
        self.server_id = server_id
        #self.queue = simpy.Resource(env, capacity=1)
        self.queue = simpy.PriorityResource(env, capacity=1)

        self.processing_frequency = processing_frequency  # fn(t)
        # Normal-environment transient fault arrival rate λ_j^0, unit: 1/s.
        # ``failure_rate`` remains a compatibility alias for older analysis
        # code; formal runtime code uses ``base_failure_rate``.
        try:
            self.base_failure_rate = float(base_failure_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError("base_failure_rate must be a finite non-negative number") from exc
        if not math.isfinite(self.base_failure_rate) or self.base_failure_rate < 0.0:
            raise ValueError("base_failure_rate must be a finite non-negative number")
        self.failure_rate = self.base_failure_rate
        try:
            self.uplink_rate_mbps = float(uplink_rate_mbps)
        except (TypeError, ValueError) as exc:
            raise ValueError("uplink_rate_mbps must be a finite positive number") from exc
        if not math.isfinite(self.uplink_rate_mbps) or self.uplink_rate_mbps <= 0.0:
            raise ValueError("uplink_rate_mbps must be a finite positive number")
        # Short alias used by the task communication helper.
        self.uplink_rate = self.uplink_rate_mbps
        self.latitude = self._validate_coordinate(latitude, "latitude", -90.0, 90.0)
        self.longitude = self._validate_coordinate(longitude, "longitude", -180.0, 180.0)

    @staticmethod
    def _validate_coordinate(value, name, minimum, maximum):
        try:
            coordinate = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc

        if not math.isfinite(coordinate):
            raise ValueError(f"{name} must be a finite number")
        if not minimum <= coordinate <= maximum:
            raise ValueError(
                f"{name} must be in [{minimum}, {maximum}], got {coordinate}"
            )
        return coordinate
