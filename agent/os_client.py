"""
Single OpenSearch client factory — used by all agent tools.
Reads connection settings from environment / .env file.
One place to change credentials, one place to change SSL settings.
"""

import os
from opensearchpy import AsyncOpenSearch
from dotenv import load_dotenv

load_dotenv()


def client() -> AsyncOpenSearch:
    """Return a configured AsyncOpenSearch client."""
    return AsyncOpenSearch(
        hosts=[os.getenv("OPENSEARCH_URL", "https://localhost:9200")],
        http_auth=(
            os.getenv("OPENSEARCH_USER", "admin"),
            os.getenv("OPENSEARCH_PASSWORD", "Pharma@2024!"),
        ),
        use_ssl=True,
        verify_certs=False,   # self-signed cert — acceptable for local dev
        ssl_show_warn=False,
    )


# Convenience alias used by ingestion scripts
INDEX = os.getenv("OPENSEARCH_INDEX", "faers_reports")
