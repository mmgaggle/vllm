# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connection configuration for the object store secondary tier."""

from dataclasses import dataclass, field


@dataclass
class ObjStoreConfig:
    """Connection parameters for an object store backend.

    When ``access_key`` and ``secret_key`` are left empty the NIXL OBJ
    plugin falls back to the AWS SDK default credential provider chain
    (IAM roles, environment variables, credential files, etc.), which
    enables workload-identity based auth on Kubernetes.

    ``nixl_params`` passes further NIXL OBJ backend parameters through as
    they are, for example the S3-over-RDMA settings (``accelerated``,
    ``rdma_transport``, ``ofi_provider``). A key that one of the fields above
    sets is refused, so that each setting has one source.
    """

    bucket: str
    endpoint_override: str
    access_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    session_token: str = field(default="", repr=False)
    region: str = ""
    scheme: str = "http"
    ca_bundle: str = ""
    nixl_params: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reserved = {
            "bucket",
            "endpoint_override",
            "scheme",
            "access_key",
            "secret_key",
            "session_token",
            "region",
            "ca_bundle",
        }
        clashes = sorted(reserved.intersection(self.nixl_params))
        if clashes:
            raise ValueError(
                f"nixl_params must not set {clashes}; use the store_config "
                "fields of the same name"
            )

    def to_nixl_params(self) -> dict[str, str]:
        """Build the NIXL backend params dict.

        Credential and optional fields are only included when non-empty
        so that the AWS SDK default credential chain can activate.
        """
        params: dict[str, str] = {
            "bucket": self.bucket,
            "endpoint_override": self.endpoint_override,
            "scheme": self.scheme,
        }
        # Omit empty optional fields so the NIXL OBJ plugin's underlying
        # AWS SDK can fall back to its default credential provider chain
        # (IAM roles, env vars, credential files, etc.).
        # https://github.com/ai-dynamo/nixl/blob/main/src/plugins/obj/README.md
        for key in ("access_key", "secret_key", "session_token", "region", "ca_bundle"):
            value = getattr(self, key)
            if value:
                params[key] = value
        # NIXL backend parameters are strings, also booleans and numbers.
        for key, value in self.nixl_params.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            params[key] = str(value)
        return params
