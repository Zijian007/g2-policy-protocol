"""Transport implementations for the public G2 wire protocol.

Use client.PolicyClient for synchronous requests and server.PolicyServer for
backend sessions. images prepares outgoing frames and decodes incoming
observations in the model worker; codec handles frame formats. Message meaning
and cross-field rules remain in protocol.
"""
