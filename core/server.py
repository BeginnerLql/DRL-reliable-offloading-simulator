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
        failure_rate,
        latitude,
        longitude,
    ):
        self.env = env
        self.server_type = server_type
        self.server_id = server_id
        #self.queue = simpy.Resource(env, capacity=1)
        self.queue = simpy.PriorityResource(env, capacity=1)

        self.processing_frequency = processing_frequency  # fn(t)
        # Observable estimated transient server-fault arrival rate λ_n, unit: 1/s.
        # A transient fault can fail the current task replica without
        # permanently disabling this server.
        self.failure_rate = failure_rate
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
