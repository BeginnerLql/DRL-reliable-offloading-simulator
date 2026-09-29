# server.py
import math

import simpy


class Server:
    def __init__(
        self,
        env,
        server_id,
        site_id,
        processing_frequency,
        base_failure_rate,
        latitude,
        longitude,
    ):
        self.env = env
        self.server_id = int(server_id)
        self.site_id = str(site_id)
        self.queue = simpy.PriorityResource(env, capacity=1)
        self.processing_frequency = float(processing_frequency)
        self.base_failure_rate = float(base_failure_rate)
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
