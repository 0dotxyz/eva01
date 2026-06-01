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
            RpcError::RpcRequestError(msg) | RpcError::ForUser(msg) => msg.contains("too large"),
            _ => false,
        },
        _ => false,
    }
}
