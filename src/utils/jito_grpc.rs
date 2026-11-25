//! Jito's block-engine searcher API over gRPC. Bundles go out authenticated by a whitelisted key
//! (`JITO_AUTH_KEYPAIR`, rate-limited apart from the JSON-RPC uuid), and the block engine streams
//! back what became of each one sent this way: forwarded to a leader, rejected and why (outbid,
//! failed simulation), processed at a slot, or dropped. Its JSON-RPC reports none of the reasons.
//!
//! One background task keeps the authenticated connection and the result stream, reconnecting
//! with backoff. While it is down, [`JitoGrpc::send_bundle`] answers [`GrpcSend::NotSent`] and the
//! bundle goes over JSON-RPC instead.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime},
};

use anyhow::{anyhow, Result};
use log::{debug, info, warn};
use solana_keypair::Keypair;
use solana_signer::Signer;
use tokio::{runtime::Runtime, sync::Notify};
use tonic::{
    codec::Streaming,
    codegen::http::uri::PathAndQuery,
    metadata::{Ascii, MetadataValue},
    transport::{Channel, ClientTlsConfig, Endpoint},
    Code, Request, Status,
};

use super::jito::InflightStatus;

/// The block engine's global endpoint, which routes to the nearest region.
pub const DEFAULT_GRPC_URL: &str = "https://mainnet.block-engine.jito.wtf";

const GENERATE_AUTH_CHALLENGE: &str = "/auth.AuthService/GenerateAuthChallenge";
const GENERATE_AUTH_TOKENS: &str = "/auth.AuthService/GenerateAuthTokens";
const SEND_BUNDLE: &str = "/searcher.SearcherService/SendBundle";
const SUBSCRIBE_BUNDLE_RESULTS: &str = "/searcher.SearcherService/SubscribeBundleResults";

/// `auth.Role.SEARCHER`.
const ROLE_SEARCHER: i32 = 1;

/// How `unary` reports a channel with no connection to send on.
const NOT_READY: &str = "not ready: ";

/// A connection (TCP and TLS) not made by then is retried later.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Authenticating and subscribing take a round trip each.
const SETUP_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a bundle's send may take: the block engine answered in 15–18 ms (measured). One that
/// times out is taken as possibly sent, and goes over JSON-RPC too.
const SEND_TIMEOUT: Duration = Duration::from_millis(500);
/// Sends in a row without a usable answer that end the session.
const FAILURES_BROKEN: u32 = 2;
/// A token renewal that failed is tried again this much later, the old token still in use.
const RENEW_RETRY: Duration = Duration::from_secs(10);
/// Failed renewals of a token that states no expiry before the session starts over.
const RENEW_TRIES: u32 = 6;
/// An HTTP/2 ping this often; a connection that doesn't answer within the timeout is dropped, and
/// with it the session, so sends go over JSON-RPC until it is back. Not more often: servers may
/// close a connection that pings too much.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
const KEEPALIVE_TIMEOUT: Duration = Duration::from_secs(10);
const TCP_KEEPALIVE: Duration = Duration::from_secs(15);
/// Reconnecting waits this long at first, doubling up to the most.
const RECONNECT_MIN: Duration = Duration::from_secs(1);
const RECONNECT_MAX: Duration = Duration::from_secs(30);
/// A session that lasted this long was healthy: the next reconnect starts from the shortest wait.
const HEALTHY_SESSION: Duration = Duration::from_secs(60);
/// The access token is renewed this long before it expires, or this often when it states no expiry.
const RENEW_BEFORE_EXPIRY: Duration = Duration::from_secs(120);
const RENEW_DEFAULT: Duration = Duration::from_secs(10 * 60);
const RENEW_MIN: Duration = Duration::from_secs(30);
/// A bundle's results are kept this long: past the longest a bundle is watched.
const RESULT_TTL: Duration = Duration::from_secs(120);
/// A bundle whose simulation failed and that was forwarded nowhere for this long won't land; a
/// copy forwarded meanwhile keeps it alive.
const REJECTED_GRACE: Duration = Duration::from_secs(1);
/// Old results are dropped at most this often.
const PRUNE_INTERVAL: Duration = Duration::from_secs(1);

/// Jito's protobuf messages (`mev-protos`), the fields eva uses.
mod proto {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct GenerateAuthChallengeRequest {
        #[prost(int32, tag = "1")]
        pub role: i32,
        #[prost(bytes = "vec", tag = "2")]
        pub pubkey: Vec<u8>,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct GenerateAuthChallengeResponse {
        #[prost(string, tag = "1")]
        pub challenge: String,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct GenerateAuthTokensRequest {
        #[prost(string, tag = "1")]
        pub challenge: String,
        #[prost(bytes = "vec", tag = "2")]
        pub client_pubkey: Vec<u8>,
        #[prost(bytes = "vec", tag = "3")]
        pub signed_challenge: Vec<u8>,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Token {
        #[prost(string, tag = "1")]
        pub value: String,
        #[prost(message, optional, tag = "2")]
        pub expires_at_utc: Option<prost_types::Timestamp>,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct GenerateAuthTokensResponse {
        #[prost(message, optional, tag = "1")]
        pub access_token: Option<Token>,
        #[prost(message, optional, tag = "2")]
        pub refresh_token: Option<Token>,
    }

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Meta {
        #[prost(uint64, tag = "1")]
        pub size: u64,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Packet {
        #[prost(bytes = "vec", tag = "1")]
        pub data: Vec<u8>,
        #[prost(message, optional, tag = "2")]
        pub meta: Option<Meta>,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Bundle {
        #[prost(message, repeated, tag = "3")]
        pub packets: Vec<Packet>,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct SendBundleRequest {
        #[prost(message, optional, tag = "1")]
        pub bundle: Option<Bundle>,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct SendBundleResponse {
        #[prost(string, tag = "1")]
        pub uuid: String,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct SubscribeBundleResultsRequest {}

    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct BundleResult {
        #[prost(string, tag = "1")]
        pub bundle_id: String,
        #[prost(oneof = "BundleResultKind", tags = "2, 3, 4, 5, 6")]
        pub result: Option<BundleResultKind>,
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum BundleResultKind {
        /// Forwarded to a leader; one per leader it went to.
        #[prost(message, tag = "2")]
        Accepted(Accepted),
        /// Not forwarded to any leader.
        #[prost(message, tag = "3")]
        Rejected(Rejected),
        #[prost(message, tag = "4")]
        Finalized(Finalized),
        #[prost(message, tag = "5")]
        Processed(Processed),
        /// Forwarded, and never landed.
        #[prost(message, tag = "6")]
        Dropped(Dropped),
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Accepted {
        #[prost(uint64, tag = "1")]
        pub slot: u64,
        #[prost(string, tag = "2")]
        pub validator_identity: String,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Rejected {
        #[prost(oneof = "RejectedReason", tags = "1, 2, 3, 4, 5")]
        pub reason: Option<RejectedReason>,
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum RejectedReason {
        /// Outbid within its state auction.
        #[prost(message, tag = "1")]
        StateAuctionBidRejected(AuctionBidRejected),
        /// Won its state auction, outbid by the other winners.
        #[prost(message, tag = "2")]
        WinningBatchBidRejected(AuctionBidRejected),
        #[prost(message, tag = "3")]
        SimulationFailure(SimulationFailure),
        #[prost(message, tag = "4")]
        InternalError(Message),
        #[prost(message, tag = "5")]
        DroppedBundle(Message),
    }
    /// `StateAuctionBidRejected` and `WinningBatchBidRejected`, which share their fields.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct AuctionBidRejected {
        #[prost(string, tag = "1")]
        pub auction_id: String,
        #[prost(uint64, tag = "2")]
        pub simulated_bid_lamports: u64,
        #[prost(string, optional, tag = "3")]
        pub msg: Option<String>,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct SimulationFailure {
        #[prost(string, tag = "1")]
        pub tx_signature: String,
        #[prost(string, optional, tag = "2")]
        pub msg: Option<String>,
    }
    /// `InternalError` and `DroppedBundle`, which share their one field.
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Message {
        #[prost(string, tag = "1")]
        pub msg: String,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Finalized {}
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Processed {
        #[prost(string, tag = "1")]
        pub validator_identity: String,
        #[prost(uint64, tag = "2")]
        pub slot: u64,
        #[prost(uint64, tag = "3")]
        pub bundle_index: u64,
    }
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct Dropped {
        /// `DroppedReason`: blockhash expired (0), partially processed (1), not finalized (2).
        #[prost(int32, tag = "1")]
        pub reason: i32,
    }
}

/// The block engine's answer to a bundle sent over gRPC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrpcSend {
    /// Taken for the auction, with its id.
    Accepted(String),
    /// Refused as a bundle (invalid, say): JSON-RPC would refuse it too.
    Rejected(String),
    /// No usable answer after it may have gone out (a timeout, a reset connection): it may land.
    Uncertain(String),
    /// Certainly not taken: no connection to send on, or the token or the rate limit refused.
    /// JSON-RPC may send it.
    NotSent(String),
    /// No session: nothing went out.
    NoSession,
}

/// What a send says of the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Health {
    /// The block engine answered about the bundle.
    Answering,
    /// No answer, or no connection: two in a row end the session.
    Failing,
    /// The token was refused: the session ends.
    Broken,
    /// Refused for the rate limit, say: nothing about the connection.
    Neutral,
}

/// What the stream told about one bundle, as far as landing goes.
struct Outcome {
    /// Forwarded to a leader at least once: a rejection elsewhere doesn't stop that copy.
    forwarded: bool,
    /// When its simulation last failed: the block engine drops such a bundle (measured, one
    /// result and nothing after). Outbid and other rejections aren't counted: a later auction may
    /// still take it.
    simulation_failed_at: Option<Instant>,
    /// Forwarded and never landed: blockhash expired, or not finalized. A later
    /// landing clears it, as a later drop clears a landing: the last word stands.
    dropped: bool,
    /// Processed (at this slot) or finalized.
    landed: Option<Option<u64>>,
    /// Rejections seen; only the first is logged at info.
    rejections: u32,
    at: Instant,
}

impl Outcome {
    fn new() -> Self {
        Self {
            forwarded: false,
            simulation_failed_at: None,
            dropped: false,
            landed: None,
            rejections: 0,
            at: Instant::now(),
        }
    }
}

#[derive(Default)]
struct Outcomes {
    by_id: HashMap<String, Outcome>,
    pruned_at: Option<Instant>,
}

#[derive(Clone)]
struct Session {
    channel: Channel,
    authorization: MetadataValue<Ascii>,
    /// Told when sends find this session broken, so the task starts over at once. Its own, so a
    /// late word about an old session never ends a new one.
    broken: Arc<Notify>,
    /// Sends in a row that got no usable answer.
    failures: Arc<AtomicU32>,
}

#[derive(Default)]
struct Shared {
    /// The authenticated connection; `None` while connecting.
    session: Mutex<Option<Session>>,
    outcomes: Mutex<Outcomes>,
}

pub struct JitoGrpc {
    runtime: Runtime,
    shared: Arc<Shared>,
}

impl Outcomes {
    /// Drops what is past [`RESULT_TTL`], at most every [`PRUNE_INTERVAL`].
    fn prune(&mut self, now: Instant) {
        if self
            .pruned_at
            .is_none_or(|at| now.saturating_duration_since(at) >= PRUNE_INTERVAL)
        {
            self.by_id
                .retain(|_, outcome| now.saturating_duration_since(outcome.at) < RESULT_TTL);
            self.pruned_at = Some(now);
        }
    }
}

impl JitoGrpc {
    /// A client whose session is up on a connection nothing listens to: sends fail to connect.
    #[cfg(test)]
    pub fn up_for_test() -> Self {
        let grpc = Self {
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap(),
            shared: Arc::new(Shared::default()),
        };
        let channel = grpc
            .runtime
            .block_on(async { Endpoint::from_static("http://127.0.0.1:1").connect_lazy() });
        *lock(&grpc.shared.session) = Some(Session {
            channel,
            authorization: "Bearer x".parse().unwrap(),
            broken: Arc::new(Notify::new()),
            failures: Arc::new(AtomicU32::new(0)),
        });
        grpc
    }

    /// Connects in the background; sends answer [`GrpcSend::NoSession`] until it is up.
    pub fn new(url: String, keypair: Keypair) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("jito-grpc")
            .enable_all()
            .build()?;
        let shared = Arc::new(Shared::default());
        info!("Jito gRPC: connecting to {url} as {}", keypair.pubkey());
        runtime.spawn(run(url, Arc::new(keypair), Arc::clone(&shared)));
        Ok(Self { runtime, shared })
    }

    /// Sends `packets` (each a transaction's wire bytes, in order) as one bundle. Never waits for a
    /// connection: without one it answers [`GrpcSend::NoSession`] at once.
    pub fn send_bundle(&self, packets: Vec<Vec<u8>>) -> GrpcSend {
        let Some(session) = lock(&self.shared.session).clone() else {
            return GrpcSend::NoSession;
        };
        let request = proto::SendBundleRequest {
            bundle: Some(proto::Bundle {
                packets: packets
                    .into_iter()
                    .map(|data| proto::Packet {
                        meta: Some(proto::Meta {
                            size: data.len() as u64,
                        }),
                        data,
                    })
                    .collect(),
            }),
        };
        // The timeout is made inside the runtime, whose timer it needs.
        let answer = self.runtime.handle().block_on(async {
            tokio::time::timeout(
                SEND_TIMEOUT,
                unary::<_, proto::SendBundleResponse>(
                    &session.channel,
                    SEND_BUNDLE,
                    request,
                    Some(&session.authorization),
                ),
            )
            .await
        });
        let (send, health) = match answer {
            Ok(Ok(response)) => {
                self.sent(&response.uuid);
                (GrpcSend::Accepted(response.uuid), Health::Answering)
            }
            Ok(Err(status)) => classify(&status),
            Err(_) => (
                GrpcSend::Uncertain("no answer in time".to_string()),
                Health::Failing,
            ),
        };
        let broken = match health {
            Health::Answering => {
                session.failures.store(0, Ordering::Relaxed);
                false
            }
            Health::Failing => {
                session.failures.fetch_add(1, Ordering::Relaxed) + 1 >= FAILURES_BROKEN
            }
            Health::Broken => true,
            Health::Neutral => false,
        };
        // The token refused, or sends failing: the task starts over, and sends go over JSON-RPC
        // meanwhile.
        if broken {
            let mut current = lock(&self.shared.session);
            if current
                .as_ref()
                .is_some_and(|current| Arc::ptr_eq(&current.broken, &session.broken))
            {
                current.take();
            }
            drop(current);
            session.broken.notify_one();
        }
        send
    }

    /// Whether an authenticated session is up.
    pub fn is_up(&self) -> bool {
        lock(&self.shared.session).is_some()
    }

    /// Notes bundle `id` as sent this way (its results may already be in).
    fn sent(&self, id: &str) {
        let mut outcomes = lock(&self.shared.outcomes);
        outcomes.prune(Instant::now());
        outcomes
            .by_id
            .entry(id.to_string())
            .or_insert_with(Outcome::new);
    }

    /// `None` when the stream knows nothing of bundle `id` (it didn't go out over gRPC). Otherwise
    /// what the stream said, when it is decisive: landed; dropped after being forwarded, which no
    /// copy survives; or its simulation failed and it was forwarded nowhere for [`REJECTED_GRACE`].
    pub fn status(&self, id: &str) -> Option<Option<InflightStatus>> {
        let outcomes = lock(&self.shared.outcomes);
        let outcome = outcomes.by_id.get(id)?;
        let failed_simulation = !outcome.forwarded
            && outcome
                .simulation_failed_at
                .is_some_and(|at| at.elapsed() >= REJECTED_GRACE);
        // A landing stands unless a drop came after it.
        Some(if outcome.dropped {
            Some(InflightStatus::Failed)
        } else if let Some(slot) = outcome.landed {
            Some(InflightStatus::Landed(slot))
        } else {
            failed_simulation.then_some(InflightStatus::Failed)
        })
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether `status` is a connection that was never made: tonic's connect error is in its sources.
fn never_connected(status: &Status) -> bool {
    let mut source = std::error::Error::source(status);
    while let Some(error) = source {
        if error.is::<tonic::ConnectError>() {
            return true;
        }
        source = error.source();
    }
    status.message().starts_with(NOT_READY)
}

/// What a refused gRPC call means for the bundle and for the session.
fn classify(status: &Status) -> (GrpcSend, Health) {
    let message = format!("{:?}: {}", status.code(), status.message());
    if never_connected(status) {
        return (GrpcSend::NotSent(message), Health::Failing);
    }
    match status.code() {
        // Refused before the bundle was looked at: the token, or a server that doesn't know the
        // call. A new session may do better.
        Code::Unauthenticated | Code::PermissionDenied | Code::Unimplemented => {
            (GrpcSend::NotSent(message), Health::Broken)
        }
        Code::ResourceExhausted | Code::NotFound => (GrpcSend::NotSent(message), Health::Neutral),
        // This bundle is already in: it may land.
        Code::AlreadyExists => (GrpcSend::Uncertain(message), Health::Answering),
        // An answer about this bundle.
        Code::InvalidArgument | Code::FailedPrecondition | Code::OutOfRange => {
            if status
                .message()
                .to_lowercase()
                .contains("already processed")
            {
                (GrpcSend::Accepted("0x0".to_string()), Health::Answering)
            } else {
                (GrpcSend::Rejected(message), Health::Answering)
            }
        }
        // The connection or the server failing (an h2 reset reads as Internal): it may have
        // arrived.
        _ => (GrpcSend::Uncertain(message), Health::Failing),
    }
}

/// One unary call on `channel`, with the session's `authorization` when given.
async fn unary<Req, Resp>(
    channel: &Channel,
    path: &'static str,
    message: Req,
    authorization: Option<&MetadataValue<Ascii>>,
) -> std::result::Result<Resp, Status>
where
    Req: prost::Message + Send + Sync + 'static,
    Resp: prost::Message + Default + Send + Sync + 'static,
{
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready()
        .await
        .map_err(|e| Status::unavailable(format!("{NOT_READY}{e}")))?;
    let mut request = Request::new(message);
    if let Some(authorization) = authorization {
        request
            .metadata_mut()
            .insert("authorization", authorization.clone());
    }
    grpc.unary(
        request,
        PathAndQuery::from_static(path),
        tonic_prost::ProstCodec::default(),
    )
    .await
    .map(tonic::Response::into_inner)
}

/// An access token for `keypair`: the block engine's challenge, prefixed with the pubkey and signed.
/// With when it expires, as an instant.
async fn authenticate(
    channel: &Channel,
    keypair: &Keypair,
) -> Result<(MetadataValue<Ascii>, Option<Instant>)> {
    let pubkey = keypair.pubkey();
    let challenge: proto::GenerateAuthChallengeResponse = unary(
        channel,
        GENERATE_AUTH_CHALLENGE,
        proto::GenerateAuthChallengeRequest {
            role: ROLE_SEARCHER,
            pubkey: pubkey.to_bytes().to_vec(),
        },
        None,
    )
    .await
    .map_err(|e| anyhow!("auth challenge refused: {e}"))?;
    let challenge = format!("{pubkey}-{}", challenge.challenge);
    let signed_challenge = keypair.sign_message(challenge.as_bytes()).as_ref().to_vec();
    let tokens: proto::GenerateAuthTokensResponse = unary(
        channel,
        GENERATE_AUTH_TOKENS,
        proto::GenerateAuthTokensRequest {
            challenge,
            client_pubkey: pubkey.to_bytes().to_vec(),
            signed_challenge,
        },
        None,
    )
    .await
    .map_err(|e| anyhow!("auth tokens refused: {e}"))?;
    let access = tokens
        .access_token
        .ok_or_else(|| anyhow!("no access token in the answer"))?;
    let authorization = format!("Bearer {}", access.value)
        .parse()
        .map_err(|e| anyhow!("unusable access token: {e}"))?;
    // At least a renewal's wait away, so a clock ahead of the block engine's can't make a token
    // look expired the moment it is issued.
    let expires = access
        .expires_at_utc
        .and_then(|at| SystemTime::try_from(at).ok())
        .map(|at| {
            Instant::now()
                + at.duration_since(SystemTime::now())
                    .unwrap_or_default()
                    .max(RENEW_MIN + RENEW_BEFORE_EXPIRY)
        });
    Ok((authorization, expires))
}

async fn subscribe(
    channel: &Channel,
    authorization: &MetadataValue<Ascii>,
) -> Result<Streaming<proto::BundleResult>> {
    let mut grpc = tonic::client::Grpc::new(channel.clone());
    grpc.ready().await.map_err(|e| anyhow!("not ready: {e}"))?;
    let mut request = Request::new(proto::SubscribeBundleResultsRequest {});
    request
        .metadata_mut()
        .insert("authorization", authorization.clone());
    Ok(grpc
        .server_streaming(
            request,
            PathAndQuery::from_static(SUBSCRIBE_BUNDLE_RESULTS),
            tonic_prost::ProstCodec::default(),
        )
        .await
        .map_err(|e| anyhow!("bundle results refused: {e}"))?
        .into_inner())
}

/// When to renew a token expiring at `expires`: never sooner than [`RENEW_MIN`], so a clock ahead
/// of the block engine's can't make it re-authenticate in a loop.
fn renew_at(expires: Option<Instant>) -> tokio::time::Instant {
    let now = Instant::now();
    let wait = expires.map_or(RENEW_DEFAULT, |at| {
        at.saturating_duration_since(now)
            .saturating_sub(RENEW_BEFORE_EXPIRY)
            .max(RENEW_MIN)
    });
    tokio::time::Instant::now() + wait
}

/// Keeps a session up, for good. Each runs as a task of its own, so even a panic in one only
/// starts the next.
async fn run(url: String, keypair: Arc<Keypair>, shared: Arc<Shared>) {
    let mut wait = RECONNECT_MIN;
    loop {
        let started = Instant::now();
        let task = tokio::spawn(session(
            url.clone(),
            Arc::clone(&keypair),
            Arc::clone(&shared),
        ));
        let e = match task.await {
            Ok(Err(e)) => e,
            Ok(Ok(never)) => match never {},
            Err(e) => anyhow!("the session task failed: {e}"),
        };
        lock(&shared.session).take();
        if started.elapsed() >= HEALTHY_SESSION {
            wait = RECONNECT_MIN;
        }
        warn!(
            "Jito gRPC: {e:#}; sending over JSON-RPC, reconnecting in {}s",
            wait.as_secs()
        );
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(RECONNECT_MAX);
    }
}

/// Connects, authenticates and streams the bundle results until something fails, renewing the
/// token before it expires (the open stream keeps going).
async fn session(
    url: String,
    keypair: Arc<Keypair>,
    shared: Arc<Shared>,
) -> Result<std::convert::Infallible> {
    let endpoint = Endpoint::from_shared(url)?
        .tls_config(ClientTlsConfig::new().with_native_roots())?
        .connect_timeout(CONNECT_TIMEOUT)
        .tcp_keepalive(Some(TCP_KEEPALIVE))
        .http2_keep_alive_interval(KEEPALIVE_INTERVAL)
        .keep_alive_timeout(KEEPALIVE_TIMEOUT)
        .keep_alive_while_idle(true);
    // The connect timeout covers TCP only: DNS and TLS are bounded here.
    let channel = tokio::time::timeout(SETUP_TIMEOUT, endpoint.connect())
        .await
        .map_err(|_| anyhow!("connecting timed out"))?
        .map_err(|e| anyhow!("can't connect: {e}"))?;
    let (authorization, expires, mut stream) = tokio::time::timeout(SETUP_TIMEOUT, async {
        let (authorization, expires) = authenticate(&channel, &keypair).await?;
        let stream = subscribe(&channel, &authorization).await?;
        Ok::<_, anyhow::Error>((authorization, expires, stream))
    })
    .await
    .map_err(|_| anyhow!("authenticating timed out"))??;
    let broken = Arc::new(Notify::new());
    *lock(&shared.session) = Some(Session {
        channel: channel.clone(),
        authorization,
        broken: Arc::clone(&broken),
        failures: Arc::new(AtomicU32::new(0)),
    });
    info!(
        "Jito gRPC: authenticated as {}; bundles go over gRPC",
        keypair.pubkey()
    );
    // Renewed beside the stream, which is never left unread meanwhile.
    let _renewal = AbortOnDrop(tokio::spawn(renew(
        channel,
        Arc::clone(&keypair),
        Arc::clone(&shared),
        Arc::clone(&broken),
        expires,
    )));
    loop {
        tokio::select! {
            result = stream.message() => match result {
                Ok(Some(result)) => record(&shared, result),
                Ok(None) => return Err(anyhow!("the bundle-result stream ended")),
                Err(e) => return Err(anyhow!("the bundle-result stream failed: {e}")),
            },
            _ = broken.notified() => return Err(anyhow!("sends failed")),
        }
    }
}

/// A task that ends with whatever holds it.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Renews the token of the session `broken` belongs to before it expires, from `expires`. A
/// failed renewal is retried, the old token serving until it expires; a send it fails then ends
/// the session.
async fn renew(
    channel: Channel,
    keypair: Arc<Keypair>,
    shared: Arc<Shared>,
    broken: Arc<Notify>,
    expires: Option<Instant>,
) {
    let mut expires = expires;
    let mut at = renew_at(expires);
    let mut failures = 0u32;
    loop {
        tokio::time::sleep_until(at).await;
        match tokio::time::timeout(SETUP_TIMEOUT, authenticate(&channel, &keypair)).await {
            Ok(Ok((authorization, renewed))) => {
                failures = 0;
                expires = renewed;
                at = renew_at(expires);
                if let Some(session) = lock(&shared.session)
                    .as_mut()
                    .filter(|session| Arc::ptr_eq(&session.broken, &broken))
                {
                    session.authorization = authorization;
                }
            }
            failed => {
                let e = match failed {
                    Ok(Err(e)) => e,
                    _ => anyhow!("timed out"),
                };
                // Past its expiry the token is refused, and one stating none is given a few tries:
                // the session starts over.
                failures += 1;
                if expires.map_or(failures >= RENEW_TRIES, |at| Instant::now() >= at) {
                    warn!("Jito gRPC: renewing the token failed ({e:#}) and it has expired");
                    broken.notify_one();
                    return;
                }
                warn!("Jito gRPC: renewing the token failed ({e:#}); retrying");
                at = tokio::time::Instant::now() + RENEW_RETRY;
            }
        }
    }
}

/// Records and logs one result from the stream.
fn record(shared: &Shared, result: proto::BundleResult) {
    use proto::{BundleResultKind as Kind, RejectedReason as Reason};
    let id = result.bundle_id;
    let now = Instant::now();
    let what = {
        let mut outcomes = lock(&shared.outcomes);
        outcomes.prune(now);
        let outcome = outcomes
            .by_id
            .entry(id.clone())
            .or_insert_with(Outcome::new);
        match result.result {
            Some(Kind::Accepted(accepted)) => {
                outcome.forwarded = true;
                // One per leader it goes to: not worth a line each in production.
                debug!(
                    "JITO_BUNDLE_RESULT bundle_id={id} forwarded slot={} leader={}",
                    accepted.slot, accepted.validator_identity
                );
                return;
            }
            Some(Kind::Rejected(rejected)) => {
                outcome.rejections += 1;
                if matches!(rejected.reason, Some(Reason::SimulationFailure(_))) {
                    outcome.simulation_failed_at = Some(now);
                }
                let why = match rejected.reason {
                    Some(Reason::StateAuctionBidRejected(bid)) => format!(
                    "rejected reason=outbid_in_state_auction bid_lamports={} auction={} msg={:?}",
                    bid.simulated_bid_lamports, bid.auction_id, bid.msg
                ),
                    Some(Reason::WinningBatchBidRejected(bid)) => format!(
                        "rejected reason=outbid_among_winners bid_lamports={} auction={} msg={:?}",
                        bid.simulated_bid_lamports, bid.auction_id, bid.msg
                    ),
                    Some(Reason::SimulationFailure(failure)) => format!(
                        "rejected reason=simulation_failure tx={} msg={:?}",
                        failure.tx_signature, failure.msg
                    ),
                    Some(Reason::InternalError(error)) => {
                        format!("rejected reason=internal_error msg={:?}", error.msg)
                    }
                    Some(Reason::DroppedBundle(dropped)) => {
                        format!("rejected reason=dropped msg={:?}", dropped.msg)
                    }
                    None => "rejected reason=unstated".to_string(),
                };
                // Possibly one per auction lost: the first says why.
                if outcome.rejections > 1 {
                    debug!("JITO_BUNDLE_RESULT bundle_id={id} {why}");
                    return;
                }
                why
            }
            Some(Kind::Processed(processed)) => {
                outcome.landed = Some(Some(processed.slot));
                outcome.dropped = false;
                format!(
                    "processed slot={} leader={} index={}",
                    processed.slot, processed.validator_identity, processed.bundle_index
                )
            }
            Some(Kind::Finalized(_)) => {
                outcome.landed.get_or_insert(None);
                outcome.dropped = false;
                // Long after the watch: not worth a line each in production.
                debug!("JITO_BUNDLE_RESULT bundle_id={id} finalized");
                return;
            }
            Some(Kind::Dropped(dropped)) => {
                // Partly processed: some of its transactions landed, which another copy sharing
                // them (Sender's) does with the liquidation. Not a verdict; the chain says.
                if dropped.reason != 1 {
                    outcome.dropped = true;
                    outcome.landed = None;
                }
                format!(
                    "dropped reason={}",
                    match dropped.reason {
                        0 => "blockhash_expired",
                        1 => "partially_processed",
                        2 => "not_finalized",
                        _ => "unknown",
                    }
                )
            }
            None => "unstated".to_string(),
        }
    };
    info!("JITO_BUNDLE_RESULT bundle_id={id} {what}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message as _;

    fn shared_with(results: Vec<proto::BundleResult>) -> Shared {
        let shared = Shared::default();
        for result in results {
            record(&shared, result);
        }
        shared
    }

    fn status(shared: Shared, id: &str) -> Option<InflightStatus> {
        let grpc = JitoGrpc {
            runtime: tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
            shared: Arc::new(shared),
        };
        grpc.status(id).flatten()
    }

    fn result(kind: proto::BundleResultKind) -> proto::BundleResult {
        proto::BundleResult {
            bundle_id: "b".to_string(),
            result: Some(kind),
        }
    }

    fn rejected() -> proto::BundleResult {
        result(proto::BundleResultKind::Rejected(proto::Rejected {
            reason: Some(proto::RejectedReason::SimulationFailure(
                proto::SimulationFailure {
                    tx_signature: "s".to_string(),
                    msg: Some("AccountNotFound".to_string()),
                },
            )),
        }))
    }

    fn accepted() -> proto::BundleResult {
        result(proto::BundleResultKind::Accepted(proto::Accepted {
            slot: 7,
            validator_identity: "v".to_string(),
        }))
    }

    #[test]
    fn a_drop_or_a_landing_settles_a_bundle_at_once() {
        // A rejection alone isn't a verdict yet (see the lost-auction test for after the grace).
        assert_eq!(status(shared_with(vec![rejected()]), "b"), None);
        assert_eq!(status(shared_with(vec![accepted()]), "b"), None);
        assert_eq!(status(shared_with(vec![]), "b"), None, "nothing said");
        let dropped = || {
            result(proto::BundleResultKind::Dropped(proto::Dropped {
                reason: 2,
            }))
        };
        assert_eq!(
            status(shared_with(vec![accepted(), rejected(), dropped()]), "b"),
            Some(InflightStatus::Failed)
        );
        // Partly processed: some of its transactions landed, as Sender's copy's do.
        let partly = result(proto::BundleResultKind::Dropped(proto::Dropped {
            reason: 1,
        }));
        assert_eq!(status(shared_with(vec![accepted(), partly]), "b"), None);
        // Processed, then dropped as not finalized: its fork is gone.
        let processed = result(proto::BundleResultKind::Processed(proto::Processed {
            validator_identity: "v".to_string(),
            slot: 9,
            bundle_index: 0,
        }));
        assert_eq!(
            status(shared_with(vec![processed.clone(), dropped()]), "b"),
            Some(InflightStatus::Failed)
        );
        // The last word stands.
        assert_eq!(
            status(shared_with(vec![dropped(), processed]), "b"),
            Some(InflightStatus::Landed(Some(9)))
        );
    }

    #[test]
    fn a_processed_bundle_landed_at_its_slot() {
        let processed = result(proto::BundleResultKind::Processed(proto::Processed {
            validator_identity: "v".to_string(),
            slot: 9,
            bundle_index: 0,
        }));
        let finalized = result(proto::BundleResultKind::Finalized(proto::Finalized {}));
        assert_eq!(
            status(
                shared_with(vec![accepted(), processed.clone(), finalized.clone()]),
                "b"
            ),
            Some(InflightStatus::Landed(Some(9)))
        );
        assert_eq!(
            status(shared_with(vec![finalized, processed]), "b"),
            Some(InflightStatus::Landed(Some(9)))
        );
    }

    #[test]
    fn refusals_are_told_apart() {
        let kind = |status: Status| classify(&status);
        assert!(matches!(
            kind(Status::unauthenticated("expired")),
            (GrpcSend::NotSent(_), Health::Broken)
        ));
        assert!(matches!(
            kind(Status::resource_exhausted("rate limited")),
            (GrpcSend::NotSent(_), Health::Neutral)
        ));
        assert!(matches!(
            kind(Status::already_exists("duplicate bundle")),
            (GrpcSend::Uncertain(_), Health::Answering)
        ));
        // A reset connection (h2 INTERNAL_ERROR) may come after the bundle arrived.
        for status in [
            Status::unavailable("reset"),
            Status::internal("h2 protocol error"),
            Status::unknown("?"),
            Status::cancelled("gone"),
        ] {
            assert!(matches!(
                kind(status),
                (GrpcSend::Uncertain(_), Health::Failing)
            ));
        }
        assert!(matches!(
            kind(Status::invalid_argument("bundle must tip")),
            (GrpcSend::Rejected(_), Health::Answering)
        ));
        assert_eq!(
            kind(Status::invalid_argument("Bundle already processed")),
            (GrpcSend::Accepted("0x0".to_string()), Health::Answering)
        );
    }

    #[test]
    fn a_connection_never_made_is_not_sent() {
        let grpc = JitoGrpc {
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap(),
            shared: Arc::new(Shared::default()),
        };
        // Nothing listens there: tonic's own connect error.
        let channel = grpc
            .runtime
            .block_on(async { Endpoint::from_static("http://127.0.0.1:1").connect_lazy() });
        *lock(&grpc.shared.session) = Some(Session {
            channel,
            authorization: "Bearer x".parse().unwrap(),
            broken: Arc::new(Notify::new()),
            failures: Arc::new(AtomicU32::new(0)),
        });
        let send = grpc.send_bundle(vec![vec![1]]);
        assert!(matches!(send, GrpcSend::NotSent(_)), "{send:?}");
    }

    #[test]
    fn a_bundle_failing_simulation_and_forwarded_nowhere_wont_land() {
        let shared = shared_with(vec![rejected()]);
        lock(&shared.outcomes)
            .by_id
            .get_mut("b")
            .unwrap()
            .simulation_failed_at = Some(Instant::now() - REJECTED_GRACE);
        assert_eq!(status(shared, "b"), Some(InflightStatus::Failed));
        // Processed with no forward reported: landed all the same.
        let processed = result(proto::BundleResultKind::Processed(proto::Processed {
            validator_identity: "v".to_string(),
            slot: 9,
            bundle_index: 0,
        }));
        let shared = shared_with(vec![rejected(), processed]);
        lock(&shared.outcomes)
            .by_id
            .get_mut("b")
            .unwrap()
            .simulation_failed_at = Some(Instant::now() - REJECTED_GRACE);
        assert_eq!(status(shared, "b"), Some(InflightStatus::Landed(Some(9))));
        // Failed again just now: the grace runs from the last word.
        let shared = shared_with(vec![rejected()]);
        lock(&shared.outcomes)
            .by_id
            .get_mut("b")
            .unwrap()
            .simulation_failed_at = Some(Instant::now() - REJECTED_GRACE);
        record(&shared, rejected());
        assert_eq!(status(shared, "b"), None);
        // Forwarded somewhere: that copy may still land.
        let shared = shared_with(vec![accepted(), rejected()]);
        lock(&shared.outcomes)
            .by_id
            .get_mut("b")
            .unwrap()
            .simulation_failed_at = Some(Instant::now() - REJECTED_GRACE);
        assert_eq!(status(shared, "b"), None);
    }

    #[test]
    fn sends_that_keep_hanging_end_the_session() {
        let grpc = JitoGrpc {
            runtime: tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap(),
            shared: Arc::new(Shared::default()),
        };
        assert_eq!(grpc.send_bundle(vec![vec![1]]), GrpcSend::NoSession);
        // A server that takes the TCP connection and never speaks: whether the bundle got out is
        // unknowable from here, so it reads as uncertain.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let channel = grpc
            .runtime
            .block_on(async { Endpoint::from_shared(url).unwrap().connect_lazy() });
        let broken = Arc::new(Notify::new());
        *lock(&grpc.shared.session) = Some(Session {
            channel,
            authorization: "Bearer x".parse().unwrap(),
            broken: Arc::clone(&broken),
            failures: Arc::new(AtomicU32::new(0)),
        });
        let started = Instant::now();
        assert!(matches!(
            grpc.send_bundle(vec![vec![1]]),
            GrpcSend::Uncertain(_)
        ));
        assert!(started.elapsed() < SEND_TIMEOUT + Duration::from_millis(500));
        assert!(lock(&grpc.shared.session).is_some(), "one slow answer");
        assert!(matches!(
            grpc.send_bundle(vec![vec![1]]),
            GrpcSend::Uncertain(_)
        ));
        assert!(lock(&grpc.shared.session).is_none(), "JSON-RPC sends next");
        // The session's task is told.
        grpc.runtime.block_on(async {
            tokio::time::timeout(Duration::from_secs(1), broken.notified())
                .await
                .expect("notified")
        });
        drop(listener);
    }

    /// The messages decode as Jito's: a `BundleResult` encoded by hand, field by field.
    #[test]
    fn a_bundle_result_decodes_from_the_wire() {
        // bundle_id = "ab" (field 1), rejected (field 3) { simulation_failure (field 3) {
        // tx_signature = "s" (field 1) } }.
        let bytes = [
            0x0a, 0x02, b'a', b'b', 0x1a, 0x05, 0x1a, 0x03, 0x0a, 0x01, b's',
        ];
        let decoded = proto::BundleResult::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded.bundle_id, "ab");
        assert_eq!(
            decoded.result,
            Some(proto::BundleResultKind::Rejected(proto::Rejected {
                reason: Some(proto::RejectedReason::SimulationFailure(
                    proto::SimulationFailure {
                        tx_signature: "s".to_string(),
                        msg: None,
                    }
                )),
            }))
        );
        // A packet is its bytes (field 1) and their size in its meta (field 2, field 1).
        let packet = proto::Packet {
            data: vec![1, 2],
            meta: Some(proto::Meta { size: 2 }),
        };
        assert_eq!(
            packet.encode_to_vec(),
            [0x0a, 0x02, 1, 2, 0x12, 0x02, 0x08, 0x02]
        );
    }
}

/// Live: an unfunded throwaway payer's v1 bundle (a ~3 KB memo transaction and a tip) through
/// gRPC, JSON-RPC and Sender; prints each answer and what the stream says. Nothing can land.
/// `JITO_AUTH_KEYPAIR_PATH`.
#[cfg(test)]
#[test]
#[ignore = "talks to Jito and Sender"]
fn a_large_v1_bundle_reaches_every_relay() {
    use crate::transaction_limits::{v1_transaction, wire_bytes, MIN_PRIORITY_FEE_LAMPORTS};
    use crate::utils::{
        helius_sender,
        jito::{JitoClient, TipEstimator},
    };
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Info)
        .try_init();
    let rpc = solana_client::rpc_client::RpcClient::new(
        "https://api.mainnet-beta.solana.com".to_string(),
    );
    let blockhash = rpc.get_latest_blockhash().unwrap();
    let payer = Keypair::new();
    let memo: solana_pubkey::Pubkey = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
        .parse()
        .unwrap();
    let big = |tag: u8| {
        let ix = solana_instruction::Instruction {
            program_id: memo,
            accounts: vec![],
            data: vec![b'a' + tag; 2_900],
        };
        v1_transaction(
            &[&payer],
            &[ix],
            blockhash,
            200_000,
            MIN_PRIORITY_FEE_LAMPORTS,
        )
        .unwrap()
    };
    let tips = TipEstimator::new(1_000_000, 0.0);
    let jito_bundle = |tag| {
        vec![
            big(tag),
            tips.build_tip_transaction(&payer, blockhash, 10_000 + u64::from(tag))
                .unwrap(),
        ]
    };
    for (i, tx) in jito_bundle(0).iter().enumerate() {
        println!("tx {i}: {} bytes", wire_bytes(tx).unwrap().len());
    }
    // The tip alone, at no priority fee: only the unfunded payer may fail it.
    println!(
        "tip simulated: {:?}",
        rpc.simulate_transaction(&jito_bundle(3)[1])
            .map(|r| r.value.err)
    );

    let path = std::env::var("JITO_AUTH_KEYPAIR_PATH").unwrap();
    let bytes: Vec<u8> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let grpc = JitoGrpc::new(
        DEFAULT_GRPC_URL.to_string(),
        Keypair::try_from(bytes.as_slice()).unwrap(),
    )
    .unwrap();
    let up = Instant::now();
    while lock(&grpc.shared.session).is_none() && up.elapsed() < Duration::from_secs(15) {
        std::thread::sleep(Duration::from_millis(100));
    }
    println!("gRPC session up: {}", lock(&grpc.shared.session).is_some());
    let wire = |txs: &[solana_transaction::versioned::VersionedTransaction]| {
        txs.iter()
            .map(|tx| wire_bytes(tx).unwrap())
            .collect::<Vec<_>>()
    };
    let sent = Instant::now();
    let answer = grpc.send_bundle(wire(&jito_bundle(0)));
    println!("gRPC: {answer:?} in {} ms", sent.elapsed().as_millis());

    let http = JitoClient::new(None, None, None);
    let http_status = JitoClient::new(None, None, None);
    let json_rpc = http.reserve().send_bundle(&jito_bundle(1));
    println!("JSON-RPC: {json_rpc:?}");
    if let Ok(crate::utils::jito::BundleSend::Accepted(id)) = &json_rpc {
        std::thread::sleep(Duration::from_millis(1_100));
        println!(
            "JSON-RPC status of its own bundle: {:?}",
            http_status.reserve().inflight_bundle_status(id)
        );
    }

    let sender = helius_sender::HeliusSender::new(None, None);
    let sender_bundle = vec![
        big(2),
        helius_sender::build_tip_transaction(&payer, blockhash, helius_sender::MIN_TIP_LAMPORTS)
            .unwrap(),
    ];
    println!("Sender: {:?}", sender.send_bundle(&sender_bundle));

    if let GrpcSend::Accepted(id) = &answer {
        std::thread::sleep(Duration::from_secs(1));
        println!(
            "JSON-RPC status of the gRPC bundle: {:?}",
            http_status.reserve().inflight_bundle_status(id)
        );
    }
    // The same bundle over JSON-RPC: the same id?
    std::thread::sleep(Duration::from_millis(1_100));
    println!(
        "same bundle over JSON-RPC: {:?}",
        http.reserve().send_bundle(&jito_bundle(0))
    );
    // Rejected for the unfunded payer: known as sent over gRPC, nothing decisive.
    if let GrpcSend::Accepted(id) = answer {
        println!("gRPC stream: {:?}", grpc.status(&id));
    }
}

/// Live: a session comes up and its token renews (forced due now, so after [`RENEW_MIN`]).
/// `JITO_AUTH_KEYPAIR_PATH`.
#[cfg(test)]
#[test]
#[ignore = "talks to Jito"]
fn a_token_renews_live() {
    let path = std::env::var("JITO_AUTH_KEYPAIR_PATH").unwrap();
    let bytes: Vec<u8> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let keypair = Arc::new(Keypair::try_from(bytes.as_slice()).unwrap());
    let grpc = JitoGrpc::new(
        DEFAULT_GRPC_URL.to_string(),
        Keypair::try_from(bytes.as_slice()).unwrap(),
    )
    .unwrap();
    let up = Instant::now();
    while !grpc.is_up() && up.elapsed() < Duration::from_secs(15) {
        std::thread::sleep(Duration::from_millis(100));
    }
    let session = lock(&grpc.shared.session).clone().expect("session up");
    let before = session.authorization.clone();
    let _renewal = grpc.runtime.block_on(async {
        AbortOnDrop(tokio::spawn(renew(
            session.channel.clone(),
            keypair,
            Arc::clone(&grpc.shared),
            Arc::clone(&session.broken),
            Some(Instant::now()),
        )))
    });
    std::thread::sleep(RENEW_MIN + Duration::from_secs(5));
    let after = lock(&grpc.shared.session).clone().expect("still up");
    assert_ne!(after.authorization, before, "renewed");
    assert!(
        Arc::ptr_eq(&after.broken, &session.broken),
        "the same session"
    );
    println!("renewed; the session stayed up");
}
