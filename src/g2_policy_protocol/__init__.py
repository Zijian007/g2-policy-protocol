"""The shared G2 wire protocol.

Start with :mod:`g2_policy_protocol.types` for service requirements, observations
and action semantics, then :mod:`g2_policy_protocol.protocol` for their validated
message exchange. :mod:`g2_policy_protocol.runtime.server` defines the backend
interface and complete request lifecycle. Training and dataset contracts are
companion modules; the communication core does not import them.
"""
