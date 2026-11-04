# Сборка
FROM rust:1-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src src
COPY web web
COPY missions missions
COPY docs/openapi.json docs/openapi.json
COPY tests tests
COPY examples examples
RUN cargo build --release --locked --bin worldsim

# Запуск: без root, только бинарник и примеры миссий
FROM debian:stable-slim
RUN useradd --system --create-home --uid 10001 worldsim && mkdir -p /data && chown worldsim /data
COPY --from=build /src/target/release/worldsim /usr/local/bin/worldsim
COPY missions /opt/worldsim/missions
USER worldsim
ENV WORLDSIM_DATA=/data
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s CMD ["worldsim", "health", "--port", "8080"]
# Токен обязателен: без WORLDSIM_TOKEN сервер на внешнем адресе не запустится
CMD ["worldsim", "serve", "--host", "0.0.0.0", "--port", "8080", "--no-open", "--log"]
