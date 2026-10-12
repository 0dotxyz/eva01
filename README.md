
# Eva01 - the Project 0 Liquidator

## Structure
* `bin` - environment configuration template
* `idls` - IDLs for Project 0 integrations (Kamino, Juplend)
* `src` - source code
* `Dockerfile` - production (`network-mainnet`) image build
* `Dockerfile.staging` - staging (`network-staging`) image build

### Configuration
The [env.template](bin/env.template) file is a template for the required and optional environment variables that are used by Eva.

## Deployment
1. Install dependencies
    * OS librarires: `sudo apt install build-essential libssl-dev pkg-config unzip`
    * Protoc:  https://grpc.io/docs/protoc-installation/#install-pre-compiled-binaries-any-os;
    * Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
1. Clone the Git repo.
1. Create and configure your environment file by using [env.template](bin/env.template) as prototype.
1. If you run through `run-eva.sh`, copy values into `docker.prod.env` (or change `run-eva.sh` to source a different file).


## Run
1. Configure `docker.prod.env`.
1. Run from VS Code task `Run (with all checks)` or execute `bash ./run-eva.sh`.
1. Optional background run: `nohup bash ./run-eva.sh >> ~/log/liquidator.log 2>&1 &`

> Initial Loading Time
The initial loading phase can take some time, depending on your RPC. Eva will load everything needed into the state, including all Marginfi Accounts. Expect the loading time to be between 1-3 minutes depending on the RPC.

> Local Docker: Run `docker build -f <CONFIG FILE> -t eva:latest .` to build an image and `docker run --env-file docker.staging.env --rm -v <WALLET>:<WALLET> eva` to run it.

## Build Profiles

Cargo now supports network-specific build features:

* `network-mainnet` (default)
* `network-staging`

### Local builds

* Mainnet: `cargo build --release --bin eva01 --no-default-features --features network-mainnet`
* Staging: `cargo build --release --bin eva01 --no-default-features --features network-staging`

### Railway builds

Use different Dockerfiles per Railway environment:

* Production environment: leave default `Dockerfile` (or set `RAILWAY_DOCKERFILE_PATH=Dockerfile`)
* Staging environment: set `RAILWAY_DOCKERFILE_PATH=Dockerfile.staging`
