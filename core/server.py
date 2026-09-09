# server.py
import simpy

class Server:
    def __init__(self, env, server_type, server_id, processing_frequency, failure_rate):
    
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

