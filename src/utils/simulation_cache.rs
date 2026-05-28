use solana_client::{
    client_error::{ClientError, ClientErrorKind},
    rpc_request::RpcError,
};

pub fn is_tx_too_large_client(err: &ClientError) -> bool {
    match err.kind() {
        ClientErrorKind::RpcError(rpc) => match rpc {
            RpcError::RpcResponseError { code, message, .. } => {
                *code == -32602 && message.contains("too large")
            }
            RpcError::RpcRequestError(msg) | RpcError::ForUser(msg) => {
                msg.contains("too large")
            }
            _ => false,
        },
        _ => false,
    }
}

#[allow(dead_code)]
fn is_tx_too_many_account_locks_client(err: &ClientError) -> bool {
    match err.kind() {
        ClientErrorKind::RpcError(rpc) => match rpc {
            RpcError::RpcResponseError { message, .. } => {
                message.contains("TooManyAccountLocks")
                    || message
                        .to_ascii_lowercase()
                        .contains("too many account locks")
            }
            RpcError::RpcRequestError(msg) | RpcError::ForUser(msg) => {
                msg.contains("TooManyAccountLocks")
                    || msg.to_ascii_lowercase().contains("too many account locks")
            }
            _ => false,
        },
        _ => false,
    }
}

#[allow(dead_code)]
pub fn is_transient_rpc_anyhow_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        if let Some(client_error) = cause.downcast_ref::<ClientError>() {
            return is_transient_rpc_client_error(client_error);
        }

        is_transient_rpc_message(&cause.to_string())
    })
}

fn is_transient_rpc_client_error(err: &ClientError) -> bool {
    match err.kind() {
        ClientErrorKind::Io(io_err) => matches!(
            io_err.kind(),
            std::io::ErrorKind::TimedOut
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::UnexpectedEof
                | std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::WouldBlock
                | std::io::ErrorKind::Interrupted
        ),
        ClientErrorKind::RpcError(rpc_err) => match rpc_err {
            RpcError::RpcResponseError { code, message, .. } => {
                matches!(*code, 408 | 429 | 500 | 502 | 503 | 504 | -32005)
                    || is_transient_rpc_message(message)
            }
            RpcError::RpcRequestError(message) | RpcError::ForUser(message) => {
                is_transient_rpc_message(message)
            }
            RpcError::ParseError(message) => is_transient_rpc_message(message),
        },
        _ => is_transient_rpc_message(&err.to_string()),
    }
}

fn is_transient_rpc_message(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("connection closed before message completed")
        || normalized.contains("operation timed out")
        || normalized.contains("request timeout")
        || normalized.contains("408 request timeout")
        || normalized.contains("429 too many requests")
        || normalized.contains("502 bad gateway")
        || normalized.contains("503 service unavailable")
        || normalized.contains("504 gateway timeout")
        || normalized.contains("temporarily unavailable")
        || normalized.contains("connection reset")
        || normalized.contains("broken pipe")
        || normalized.contains("deadline has elapsed")
        || normalized.contains("timeout")
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn test_is_transient_rpc_anyhow_error_true_for_timeout_io() {
        let err = ClientError {
            request: None,
            kind: ClientErrorKind::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "operation timed out",
            )),
        };
        let anyhow_err = anyhow!(err);
        assert!(is_transient_rpc_anyhow_error(&anyhow_err));
    }

    #[test]
    fn test_is_transient_rpc_anyhow_error_false_for_non_transient() {
        let err = ClientError {
            request: None,
            kind: ClientErrorKind::Custom("deterministic failure".to_string()),
        };
        let anyhow_err = anyhow!(err);
        assert!(!is_transient_rpc_anyhow_error(&anyhow_err));
    }
}
