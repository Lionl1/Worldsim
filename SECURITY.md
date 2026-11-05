# Security

## Reporting a vulnerability

Use a [private message on GitHub](https://github.com/Lionl1/Worldsim/security/advisories/new), not a public issue.

## How protection works

- By default the server listens only on `127.0.0.1`.
- On an external address the server will not start without a token. The token is set through `WORLDSIM_TOKEN` and passed in the `Authorization: Bearer` header. `WORLDSIM_VIEWER_TOKEN` adds a read-only token that sees results and descriptions and cannot start calculations.
- Requests from third-party sites are rejected; the headers checked are `Origin` and `Host`. They can be allowed with the `--cors` flag.
- Request size, connection count and background job count are limited.
- Requests must not execute code: a mission description is data only.

- `worldsim loop` has no access control, so it listens only on 127.0.0.1.

## What is missing

The server has no TLS of its own: on a network put it behind a reverse proxy, such as the Caddy setup in `deploy/`. There are no user accounts, and only two roles. There has been no independent security audit. The main token gives full access.

The web interface loads three.js from `cdn.jsdelivr.net`; in a closed network the 3D tabs will not work without it.
