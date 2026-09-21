"""python -m frosty_mcp: stdio by default; optional authenticated private HTTP."""
import argparse
import hmac
import ipaddress
import os
from mcp.server.transport_security import TransportSecuritySettings
from .client import StudioClient
from .server import create_server


class BearerGuard:
    def __init__(self, app, token):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope['type']=='http' and self.token:
            headers = dict(scope.get('headers', []))
            expected = ('Bearer '+self.token).encode()
            if not hmac.compare_digest(headers.get(b'authorization', b''), expected):
                body = b'{"error":"Bearer token required"}'
                await send({'type':'http.response.start','status':401,'headers':[
                    (b'content-type',b'application/json'),(b'www-authenticate',b'Bearer'),
                    (b'content-length',str(len(body)).encode())]})
                await send({'type':'http.response.body','body':body})
                return
        await self.app(scope, receive, send)


def http_app(server, host, port, token=None):
    address = ipaddress.ip_address('127.0.0.1' if host=='localhost' else host)
    if address.is_unspecified or not (address.is_loopback or address.is_private):
        raise ValueError('Bind to loopback or a specific private/VPN address')
    if not address.is_loopback and (not token or len(token)<24):
        raise ValueError('A network listener requires FROSTY_MCP_TOKEN with at least 24 characters')
    host_header = '['+host+']' if ':' in host else host
    allowed = [host_header+':'+str(port)]
    if address.is_loopback:
        allowed += ['localhost:'+str(port),'127.0.0.1:'+str(port),'[::1]:'+str(port)]
    security = TransportSecuritySettings(enable_dns_rebinding_protection=True,
        allowed_hosts=allowed, allowed_origins=['http://'+h for h in allowed])
    app = server.streamable_http_app(host=host, stateless_http=True, json_response=True,
        max_request_body_size=36_000_000, transport_security=security)
    return BearerGuard(app, token)


def main():
    parser = argparse.ArgumentParser(description='Frosty Image + VL MCP companion (no model loading)')
    parser.add_argument('--studio-url', default=os.getenv('FROSTY_STUDIO_URL','http://127.0.0.1:8890'))
    parser.add_argument('--transport', choices=['stdio','streamable-http'], default='stdio')
    parser.add_argument('--host', default=os.getenv('FROSTY_MCP_HOST','127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.getenv('FROSTY_MCP_PORT','8891')))
    args = parser.parse_args()
    try:
        client = StudioClient(args.studio_url,
            [p for p in os.getenv('FROSTY_MCP_INPUT_DIRS','').split(os.pathsep) if p],
            token=os.getenv('FROSTY_STUDIO_TOKEN'))
        server = create_server(client)
        if args.transport=='stdio':
            server.run()
        else:
            import uvicorn
            app = http_app(server, args.host, args.port, os.getenv('FROSTY_MCP_TOKEN'))
            uvicorn.run(app, host=args.host, port=args.port, log_level='info')
    except ValueError as exc:
        parser.error(str(exc))


if __name__=='__main__':
    main()
