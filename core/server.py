import simpy


class Server:
    """Fixed server profile: CPU cycles/s, upload MB/s, base failure rate s^-1."""

    def __init__(
        self,
        env,
        server_id,
        site_id,
        processing_frequency,
        base_failure_rate,
        transmission_rate,
        latitude,
        longitude,
    ):
        self.env = env
        self.server_id = int(server_id)
        self.site_id = str(site_id)
        self.queue = simpy.PriorityResource(env, capacity=1)
        self.processing_frequency = float(processing_frequency)
        self.base_failure_rate = float(base_failure_rate)
        self.transmission_rate = float(transmission_rate)
        self.latitude = float(latitude)
        self.longitude = float(longitude)
