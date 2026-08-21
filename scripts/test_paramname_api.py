#!/usr/bin/env python3
"""Test the production paramname query API.

This script only sends a read-only GET request.  The API path is intentionally
required because the supplied information contains the server address but not
the actual endpoint path.

Example (run on the production machine):
    python scripts/test_paramname_api.py --api-path /api/<paramname-endpoint>

Or provide the complete endpoint URL:
    python scripts/test_paramname_api.py \
        --endpoint http://10.174.38.61:5000/api/<paramname-endpoint>
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_SERVER_URL = "http://10.174.38.61:5000"
DEFAULT_PARAM_KEY = "paramname"
DEFAULT_PARAM_VALUE = "capiaimachid"
DEFAULT_EXPECTED_FUNCTION = "AIDetectionByMachid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="查詢生產環境 paramname API，並驗證 capiaimachid 回應。"
    )
    parser.add_argument(
        "--endpoint",
        help="完整 API URL，例如 http://10.174.38.61:5000/api/xxx",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"服務根網址（預設：{DEFAULT_SERVER_URL}）",
    )
    parser.add_argument(
        "--api-path",
        help="API 路徑，例如 /api/xxx；未提供 --endpoint 時必填",
    )
    parser.add_argument(
        "--param-key",
        default=DEFAULT_PARAM_KEY,
        help=f"查詢參數名稱（預設：{DEFAULT_PARAM_KEY}）",
    )
    parser.add_argument(
        "--param-value",
        default=DEFAULT_PARAM_VALUE,
        help=f"查詢參數值（預設：{DEFAULT_PARAM_VALUE}）",
    )
    parser.add_argument(
        "--expected-function",
        default=DEFAULT_EXPECTED_FUNCTION,
        help=f"預期 functiondatalist（預設：{DEFAULT_EXPECTED_FUNCTION}）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout 秒數（預設：10）",
    )
    return parser.parse_args()


def build_url(args: argparse.Namespace) -> str:
    if args.endpoint:
        url = args.endpoint
    else:
        if not args.api_path:
            raise ValueError("必須提供 --endpoint，或同時提供 --server-url 與 --api-path")
        url = f"{args.server_url.rstrip('/')}/{args.api_path.lstrip('/')}"

    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"無效的 API URL：{url}")

    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query[args.param_key] = args.param_value
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def request_json(url: str, timeout: float) -> tuple[int, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "PatchCore-paramname-api-test/1.0",
        },
        method="GET",
    )
    with urlopen(request, timeout=timeout) as response:
        status = int(response.status)
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read().decode(charset)
    return status, json.loads(body)


def validate_response(payload: Any, expected_function: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["回應不是 JSON object"]

    if payload.get("code") != 200:
        errors.append(f"JSON code 不是 200，而是 {payload.get('code')!r}")
    if payload.get("type") != "query":
        errors.append(f"JSON type 不是 query，而是 {payload.get('type')!r}")

    data = payload.get("data")
    if not isinstance(data, list):
        errors.append("JSON data 不是 list")
        return errors
    if not data:
        errors.append("JSON data 為空，找不到 paramname 資料")
        return errors

    matching_records = [
        item
        for item in data
        if isinstance(item, dict)
        and item.get("functiondatalist") == expected_function
    ]
    if not matching_records:
        actual = [
            item.get("functiondatalist")
            for item in data
            if isinstance(item, dict)
        ]
        errors.append(
            f"找不到 functiondatalist={expected_function!r}；實際值：{actual!r}"
        )
        return errors

    for item in matching_records:
        function_value = item.get("functionvalue")
        if not isinstance(function_value, str) or not function_value.strip():
            errors.append("匹配資料的 functionvalue 為空或不是字串")

    return errors


def print_summary(status: int, url: str, payload: Any) -> None:
    print(f"HTTP status: {status}")
    print(f"Request URL: {url}")
    if not isinstance(payload, dict):
        return

    print(f"code: {payload.get('code')}")
    print(f"msg: {payload.get('msg')}")
    print(f"total: {payload.get('total')}")

    data = payload.get("data")
    if isinstance(data, list):
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                print(f"data[{index}]: {item!r}")
                continue
            print(
                f"data[{index}]: belong={item.get('belong')!r}, "
                f"rstation={item.get('rstation')!r}, "
                f"functiondatalist={item.get('functiondatalist')!r}, "
                f"functionvalue={item.get('functionvalue')!r}"
            )


def main() -> int:
    args = parse_args()

    try:
        url = build_url(args)
    except ValueError as exc:
        print(f"設定錯誤：{exc}", file=sys.stderr)
        return 2

    print("開始測試唯讀 paramname API。")
    print(f"param: {args.param_key}={args.param_value}")

    try:
        status, payload = request_json(url, args.timeout)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP 錯誤：{exc.code} {exc.reason}", file=sys.stderr)
        if body:
            print(body, file=sys.stderr)
        return 2
    except (URLError, TimeoutError, OSError) as exc:
        print(f"連線失敗：{exc}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"回應不是有效 JSON：{exc}", file=sys.stderr)
        return 2

    print_summary(status, url, payload)
    errors = validate_response(payload, args.expected_function)
    if status != 200:
        errors.insert(0, f"HTTP status 不是 200，而是 {status}")

    if errors:
        print("\nTEST FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("\nTEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
