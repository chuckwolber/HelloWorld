# TLS
An extension of the [Basic Threaded Client/Server](../Basic) project that uses
OpenSSL v1.1.x libraries to encrypt the connection with TLS. This is a
simplified implementation that avoids the use of BIO and other abstractions.
This is also suitable for situations that wish to reuse a socket for encrypted
and unencrypted traffic.

## Security Warning
This implementation is probably secure enough to convey the basics, but it is
missing certificate validation and likely other important things. If you use
this implementation for something important, it is on you to conduct a proper
security audit.

## Traffic Sniffing
The server side of this implementation respects the SSLKEYLOGFILE environment
variable. Set this value to a key log file in order to allow Wireshark to
unencrypt the traffic. See Wireshark documentation for configuration details.

Note: Wireshark cannot (currently) decrypt TLSv1.3 using the SSLKEYLOGFILE
method so TLS version is limited to 1.2 or lower in this code.

## Key Generation
Self signed certificates are automatically generated via the Makefile. This
approach is not terribly secure, but it is sufficient to get the point across.

If you want something more secure, check out the [LetsEncrypt
Project](https://letsencrypt.org/). Your implementation will also need to
validate certificates against a known CA (Certificate Authority).

