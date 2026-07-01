from __future__ import annotations

import argparse
import ipaddress
import os
import re
import socket
import ssl
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
CERT_DIR = BASE_DIR / ".certs"
CA_KEY_PATH = CERT_DIR / "speechpractice-local-ca.key"
CA_CERT_PATH = CERT_DIR / "speechpractice-local-ca.crt"
SERVER_KEY_PATH = CERT_DIR / "speechpractice-lan.key"
SERVER_CERT_PATH = CERT_DIR / "speechpractice-lan.crt"


class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class HTTPSRequestHandler(WSGIRequestHandler):
    def get_environ(self):
        environ = super().get_environ()
        environ["HTTPS"] = "on"
        environ["wsgi.url_scheme"] = "https"
        return environ


def private_ipv4_addresses() -> list[str]:
    addresses = {"127.0.0.1"}

    for hostname in {socket.gethostname(), socket.getfqdn()}:
        try:
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                addresses.add(info[4][0])
        except socket.gaierror:
            pass

    if os.name == "nt":
        try:
            output = subprocess.check_output(
                ["ipconfig"],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except (OSError, subprocess.CalledProcessError):
            output = ""
        for match in re.findall(r"IPv4[^\r\n:]*:\s*([0-9.]+)", output):
            addresses.add(match)

    def sort_key(value: str) -> tuple[int, str]:
        ip = ipaddress.ip_address(value)
        return (0 if ip.is_private and not ip.is_loopback else 1, value)

    return sorted(addresses, key=sort_key)


def make_private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def write_private_key(path: Path, key) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )


def ensure_local_ca():
    CERT_DIR.mkdir(exist_ok=True)
    if CA_KEY_PATH.exists() and CA_CERT_PATH.exists():
        ca_key = serialization.load_pem_private_key(CA_KEY_PATH.read_bytes(), password=None)
        ca_cert = x509.load_pem_x509_certificate(CA_CERT_PATH.read_bytes())
        return ca_key, ca_cert

    ca_key = make_private_key()
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SpeechPractice local development"),
            x509.NameAttribute(NameOID.COMMON_NAME, "SpeechPractice local development CA"),
        ]
    )
    now = datetime.now(timezone.utc)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1825))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    write_private_key(CA_KEY_PATH, ca_key)
    CA_CERT_PATH.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    return ca_key, ca_cert


def write_server_certificate(ca_key, ca_cert, hostnames: set[str], ips: set[str]) -> None:
    server_key = make_private_key()
    now = datetime.now(timezone.utc)
    san_entries = [x509.DNSName(hostname) for hostname in sorted(hostnames)]
    san_entries.extend(x509.IPAddress(ipaddress.ip_address(ip)) for ip in sorted(ips))
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SpeechPractice local development"),
            x509.NameAttribute(NameOID.COMMON_NAME, "SpeechPractice LAN"),
        ]
    )
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=False,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )
    write_private_key(SERVER_KEY_PATH, server_key)
    SERVER_CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def ensure_server_certificate(addresses: list[str]) -> None:
    ca_key, ca_cert = ensure_local_ca()
    hostnames = {"localhost", socket.gethostname()}
    ips = set(addresses)
    write_server_certificate(ca_key, ca_cert, hostnames, ips)


def build_application():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "speechpractice_web.settings")
    from django.conf import settings
    from django.core.wsgi import get_wsgi_application

    application = get_wsgi_application()
    if settings.DEBUG:
        from django.contrib.staticfiles.handlers import StaticFilesHandler

        application = StaticFilesHandler(application)
    return application


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SpeechPractice over HTTPS for LAN microphone testing.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    addresses = private_ipv4_addresses()
    ensure_server_certificate(addresses)

    application = build_application()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(SERVER_CERT_PATH, SERVER_KEY_PATH)

    with make_server(
        args.host,
        args.port,
        application,
        server_class=ThreadingWSGIServer,
        handler_class=HTTPSRequestHandler,
    ) as server:
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print("")
        print("Starting SpeechPractice HTTPS server...")
        print("")
        print("Install this certificate authority on your phone once, then trust it for VPN/apps or browser use:")
        print(f"  {CA_CERT_PATH}")
        print("")
        print("Open:")
        print(f"  https://127.0.0.1:{args.port}/")
        for address in addresses:
            if address != "127.0.0.1":
                print(f"  https://{address}:{args.port}/")
        print("")
        print("Press Ctrl+C to stop the server.")
        print("")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping HTTPS server.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
